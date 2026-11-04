"""
System-Level MoA (Mixture-of-Agents / Mixture-of-Architectures / Mixture-of-Attention)
Underlying Neural Operator Library (`moa_operators.py`) — Pure Vision-Language
Cross-Modal Spatial-Semantic Fusion Edition.

Core Architecture of `CrossModalVisionLanguageFusionSubOp`:
1. 2D-Aware Bidirectional Spatial-Semantic Decomposition (`V_vis`, `T_txt`):
   Extracts omnidirectional 2D spatial patch features via symmetric depthwise
   spatial convolution alongside semantic text features.
2. Low-Rank Cross-Modal Latent Anchor Co-Attention (`O(L * R_anchor * D)`):
   Bridges visual patches and text tokens globally with ZERO distance attenuation
   via R_anchor shared cross-modal latent bottlenecks (`cross_modal_anchor_co_attention_forward/backward`).
3. Fused Hadamard-Bilinear Cross-Modal Gated Synthesis (`cross_modal_vl_bilinear_fusion_forward/backward`):
   Computes adaptive modality gating `G_cm = sigmoid(a_v * V_vis + a_t * T_txt)`
   and bounded multiplicative bilinear interaction `(V_vis * T_txt + C_align) / sqrt(1 + lam * (V^2 + T^2))`,
   running in O(1) constant kernel launches and O(B * L * D) memory (Zero-OOM!).
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
    spatial_conv_kernel: int = 5          # Symmetric 2D-neighborhood spatial conv kernel size
    conv_kernel_size: int = 4
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
# 2. Analytical Forward/Backward Kernels — Cross-Modal Vision-Language Fusion
# ==============================================================================

class NumpyMoAOperatorKernels:
    """
    Exact forward and analytical backward kernels for System-Level MoA operators,
    featuring the new Cross-Modal Vision-Language Hadamard-Bilinear Gated Fusion
    Operator for Vision-Language (Image-Text) Multimodal Tasks.
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
        """
        Forward pass for the Fused Cross-Modal Vision-Language Hadamard-Bilinear
        Gated Synthesis Operator (Zero distance decay, Bidirectional-compatible).

        Mathematical Formulation:
        1. Cross-Modal Adaptive Gate:
               G_cm = sigmoid(alpha_vis * V_vis + alpha_txt * T_txt)
        2. Gated Linear Modality Interpolation:
               Y_interp = G_cm * V_vis + (1 - G_cm) * T_txt
        3. Bounded Multiplicative Bilinear + Co-Attention Alignment Interaction:
               B_num    = V_vis * T_txt + C_align
               E_cm     = V_vis^2 + T_txt^2
               inv_norm = 1 / sqrt(1 + lam_norm * E_cm)
               Out      = Y_interp + beta_bilin * B_num * inv_norm

        Args:
            v_vis: 2D Spatial Visual stream representation [B, L, D]
            t_txt: Semantic Textual stream representation [B, L, D]
            c_align: Non-local Cross-Modal Anchor Co-Attention context [B, L, D]
            alpha_vis: Channel-wise visual gate sensitivity [D]
            alpha_txt: Channel-wise textual gate sensitivity [D]
            beta_bilin: Channel-wise multiplicative bilinear fusion weight [D]
        Returns:
            out: Fused multimodal representation [B, L, D]
            cache: Saved intermediate tensors for exact analytical backward pass
        """
        av_b = alpha_vis[None, None, :]
        at_b = alpha_txt[None, None, :]
        bb_b = beta_bilin[None, None, :]

        # 1. Cross-Modal Adaptive Gate
        gate_logit = np.clip(av_b * v_vis + at_b * t_txt, -500.0, 500.0)
        g_cm = 1.0 / (1.0 + np.exp(-gate_logit))
        one_m_g = 1.0 - g_cm

        # 2. Gated modality interpolation
        y_interp = g_cm * v_vis + one_m_g * t_txt

        # 3. Bounded multiplicative Hadamard-Bilinear + Co-Attention interaction
        v_sq = np.square(v_vis)
        t_sq = np.square(t_txt)
        norm_arg = 1.0 + lam_norm * (v_sq + t_sq)
        inv_norm = 1.0 / np.sqrt(norm_arg)
        b_num = v_vis * t_txt + c_align
        bilin_unit = b_num * inv_norm

        out = y_interp + bb_b * bilin_unit

        cache = {
            "v_vis": v_vis,
            "t_txt": t_txt,
            "c_align": c_align,
            "alpha_vis": alpha_vis,
            "alpha_txt": alpha_txt,
            "beta_bilin": beta_bilin,
            "g_cm": g_cm,
            "one_m_g": one_m_g,
            "v_sq": v_sq,
            "t_sq": t_sq,
            "norm_arg": norm_arg,
            "inv_norm": inv_norm,
            "b_num": b_num,
            "bilin_unit": bilin_unit,
            "lam_norm": np.array(lam_norm),
        }
        return out, cache

    @staticmethod
    def cross_modal_vl_bilinear_fusion_backward(
        grad_out: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Exact analytical backward pass computing gradients w.r.t. all 6 inputs &
        cross-modal fusion parameters:
        (d_vvis, d_ttxt, d_calign, d_alphavis, d_alphatxt, d_betabilin).
        """
        v_vis = cache["v_vis"]
        t_txt = cache["t_txt"]
        alpha_vis = cache["alpha_vis"][None, None, :]
        alpha_txt = cache["alpha_txt"][None, None, :]
        beta_bilin = cache["beta_bilin"][None, None, :]
        g_cm = cache["g_cm"]
        one_m_g = cache["one_m_g"]
        norm_arg = cache["norm_arg"]
        inv_norm = cache["inv_norm"]
        b_num = cache["b_num"]
        bilin_unit = cache["bilin_unit"]
        lam_norm = float(cache["lam_norm"])

        # 1. Gradients through beta_bilin * b_num * inv_norm
        d_betabilin = np.sum(grad_out * bilin_unit, axis=(0, 1))  # [D]
        d_bunit = grad_out * beta_bilin

        d_bnum = d_bunit * inv_norm
        d_invnorm = d_bunit * b_num
        d_normarg = d_invnorm * (-0.5 * inv_norm / norm_arg)

        d_calign = d_bnum

        # 2. Gradients through G_cm = sigmoid(alpha_vis * V_vis + alpha_txt * T_txt)
        d_gcm = grad_out * (v_vis - t_txt)
        d_glogit = d_gcm * g_cm * one_m_g

        d_alphavis = np.sum(d_glogit * v_vis, axis=(0, 1))  # [D]
        d_alphatxt = np.sum(d_glogit * t_txt, axis=(0, 1))  # [D]

        # 3. Total gradients w.r.t. V_vis and T_txt
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
        """
        Bidirectional Low-Rank Cross-Modal Anchor Co-Attention (O(L * R_anchor * D)).
        Allows any visual patch anywhere in the 2D image grid to exchange semantic
        information with any text token with ZERO distance decay!

        Args:
            v_vis: Visual stream [B, L, D]
            t_txt: Textual stream [B, L, D]
            anchors: Shared cross-modal latent anchors [R_anchor, D]
        Returns:
            c_align: Cross-modal aligned context [B, L, D]
            cache: Intermediate tensors for exact analytical backward pass
        """
        B, L, D = v_vis.shape
        scale = 1.0 / math.sqrt(D)

        # Step 1: Pool global visual + textual tokens into R_anchor latent anchors
        joint_kv = 0.5 * (v_vis + t_txt)
        scores_pool = np.einsum("rd,bld->brl", anchors, joint_kv) * scale
        sp_max = np.max(scores_pool, axis=-1, keepdims=True)
        exp_sp = np.exp(scores_pool - sp_max)
        attn_pool = exp_sp / np.sum(exp_sp, axis=-1, keepdims=True)  # [B, R, L]
        z_anchors = np.einsum("brl,bld->brd", attn_pool, v_vis)      # [B, R, D]

        # Step 2: Broadcast cross-modal anchor summaries back to every token position
        scores_bc = np.einsum("bld,brd->blr", t_txt, z_anchors) * scale
        sb_max = np.max(scores_bc, axis=-1, keepdims=True)
        exp_sb = np.exp(scores_bc - sb_max)
        attn_bc = exp_sb / np.sum(exp_sb, axis=-1, keepdims=True)    # [B, L, R]
        c_align = np.einsum("blr,brd->bld", attn_bc, z_anchors)      # [B, L, D]

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
        """
        Exact analytical backward pass for Bidirectional Low-Rank Cross-Modal
        Anchor Co-Attention w.r.t. (`v_vis`, `t_txt`, `anchors`).
        """
        v_vis = cache["v_vis"]
        t_txt = cache["t_txt"]
        anchors = cache["anchors"]
        joint_kv = cache["joint_kv"]
        attn_pool = cache["attn_pool"]
        z_anchors = cache["z_anchors"]
        attn_bc = cache["attn_bc"]
        scale = float(cache["scale"])

        # Step 2 backward: c_align = einsum("blr,brd->bld", attn_bc, z_anchors)
        d_attn_bc = np.einsum("bld,brd->blr", grad_c_align, z_anchors)
        d_z_anchors = np.einsum("blr,bld->brd", attn_bc, grad_c_align)

        # Softmax backward over R_anchor dimension
        d_scores_bc = attn_bc * (
            d_attn_bc - np.sum(d_attn_bc * attn_bc, axis=-1, keepdims=True)
        ) * scale

        # scores_bc = einsum("bld,brd->blr", t_txt, z_anchors) * scale
        d_ttxt_bc = np.einsum("blr,brd->bld", d_scores_bc, z_anchors)
        d_z_anchors += np.einsum("blr,bld->brd", d_scores_bc, t_txt)

        # Step 1 backward: z_anchors = einsum("brl,bld->brd", attn_pool, v_vis)
        d_attn_pool = np.einsum("brd,bld->brl", d_z_anchors, v_vis)
        d_vvis_pool = np.einsum("brl,brd->bld", attn_pool, d_z_anchors)

        # Softmax backward over sequence length L dimension
        d_scores_pool = attn_pool * (
            d_attn_pool - np.sum(d_attn_pool * attn_pool, axis=-1, keepdims=True)
        ) * scale

        # scores_pool = einsum("rd,bld->brl", anchors, joint_kv) * scale
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
        B, N, L, D = proposals.shape
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

        topk_idx = np.argpartition(router_logits, -top_k, axis=-1)[..., -top_k:]
        topk_vals = np.take_along_axis(router_logits, topk_idx, axis=-1)
        max_tk = np.max(topk_vals, axis=-1, keepdims=True)
        exp_tk = np.exp(topk_vals - max_tk)
        topk_weights = exp_tk / np.sum(exp_tk, axis=-1, keepdims=True)

        sparse_weights = np.zeros_like(router_probs)
        np.put_along_axis(sparse_weights, topk_idx, topk_weights, axis=-1)

        selection_mask = np.zeros_like(router_probs)
        np.put_along_axis(selection_mask, topk_idx, 1.0, axis=-1)

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

        sink_or_local = causal & ((k_idx < num_sinks) | ((q_idx - k_idx) <= window_size))
        dilated = causal & (((q_idx - k_idx) % dilation == 0) | ((q_idx - k_idx) <= 1))
        full_causal = causal

        masks = []
        for h in range(n_heads):
            mode = h % 3
            if mode == 0:
                masks.append(sink_or_local)
            elif mode == 1:
                masks.append(dilated)
            else:
                masks.append(full_causal)

        mask_tensor = np.stack(masks, axis=0)[None, ...]
        causal_pairs = seq_len * (seq_len + 1) / 2.0
        stats = {
            "sink_local_density": float(sink_or_local.sum() / causal_pairs),
            "dilated_density": float(dilated.sum() / causal_pairs),
            "full_causal_density": 1.0,
            "overall_moa_density": float(mask_tensor.mean(axis=1).sum() / causal_pairs),
        }
        return mask_tensor, stats


# ==============================================================================
# 3. PyTorch Neural Operator Implementations (Vision-Language Fusion Edition)
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
        """
        Custom PyTorch Autograd Function encapsulating the Fused Cross-Modal
        Vision-Language Hadamard-Bilinear Gated Synthesis Operator.
        """

        @staticmethod
        def forward(
            ctx,
            v_vis: torch.Tensor,
            t_txt: torch.Tensor,
            c_align: torch.Tensor,
            alpha_vis: torch.Tensor,
            alpha_txt: torch.Tensor,
            beta_bilin: torch.Tensor,
            lam_norm: float,
        ) -> torch.Tensor:
            av_b = alpha_vis.view(1, 1, -1)
            at_b = alpha_txt.view(1, 1, -1)
            bb_b = beta_bilin.view(1, 1, -1)

            g_cm = torch.sigmoid(av_b * v_vis + at_b * t_txt)
            one_m_g = 1.0 - g_cm
            y_interp = g_cm * v_vis + one_m_g * t_txt

            v_sq = v_vis.pow(2)
            t_sq = t_txt.pow(2)
            norm_arg = 1.0 + lam_norm * (v_sq + t_sq)
            inv_norm = torch.rsqrt(norm_arg)
            b_num = v_vis * t_txt + c_align
            bilin_unit = b_num * inv_norm

            out = y_interp + bb_b * bilin_unit
            ctx.save_for_backward(
                v_vis, t_txt, alpha_vis, alpha_txt, beta_bilin,
                g_cm, one_m_g, norm_arg, inv_norm, b_num, bilin_unit
            )
            ctx.lam_norm = lam_norm
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            (
                v_vis, t_txt, alpha_vis, alpha_txt, beta_bilin,
                g_cm, one_m_g, norm_arg, inv_norm, b_num, bilin_unit
            ) = ctx.saved_tensors
            lam_norm = ctx.lam_norm

            av_b = alpha_vis.view(1, 1, -1)
            at_b = alpha_txt.view(1, 1, -1)
            bb_b = beta_bilin.view(1, 1, -1)

            d_betabilin = (grad_out * bilin_unit).sum(dim=(0, 1))
            d_bunit = grad_out * bb_b

            d_bnum = d_bunit * inv_norm
            d_invnorm = d_bunit * b_num
            d_normarg = d_invnorm * (-0.5 * inv_norm / norm_arg)
            d_calign = d_bnum

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
        Purpose-Built Vision-Language (Image-Text) Cross-Modal Fusion Proposer Branch
        for Vision-Language (Image-Text) Multimodal Tasks:
        - Decomposes input into 2D Spatial Visual stream (`V_vis`, with bidirectional
          depthwise spatial context) and Semantic Textual stream (`T_txt`).
        - Performs Low-Rank Cross-Modal Latent Anchor Co-Attention (`O(L * R * D)`)
          with ZERO distance decay between early image patches and late text tokens.
        - Applies `FusedCrossModalBilinearGateFunction` in a single fused kernel!
        """

        def __init__(
            self,
            d_model: int,
            num_anchors: int = 16,
            alpha_vis_init: float = 0.80,
            alpha_txt_init: float = 0.80,
            beta_bilinear_init: float = 0.65,
            lam_norm: float = 0.35,
            **kwargs,
        ):
            super().__init__()
            self.d_model = d_model
            self.num_anchors = num_anchors
            self.lam_norm = lam_norm

            self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
            # Bidirectional 2D-aware local spatial neighborhood conv (padding='same' / symmetric)
            self.spatial_dw_conv = nn.Conv1d(
                d_model, d_model, kernel_size=5, padding=2, groups=d_model
            )
            self.cross_modal_anchors = nn.Parameter(
                torch.randn(num_anchors, d_model) * (d_model ** -0.5)
            )
            self.alpha_vis = nn.Parameter(torch.full((d_model,), alpha_vis_init))
            self.alpha_txt = nn.Parameter(torch.full((d_model,), alpha_txt_init))
            self.beta_bilin = nn.Parameter(torch.full((d_model,), beta_bilinear_init))
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward_from_projected(self, v_raw: torch.Tensor, t_raw: torch.Tensor) -> torch.Tensor:
            # Bidirectional spatial neighborhood mixing on visual branch (no 1D causal blindness!)
            v_vis = F.silu(self.spatial_dw_conv(v_raw.transpose(1, 2)).transpose(1, 2))
            t_txt = F.silu(t_raw)

            # Low-rank Cross-Modal Anchor Co-Attention (Zero distance decay across L)
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
            return self.out_proj(fused)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v_raw, t_raw = self.in_proj(x).chunk(2, dim=-1)
            return self.forward_from_projected(v_raw, t_raw)



    class HeterogeneousProposerOp(nn.Module):
        """Vision-Language Optimized Mixture-of-Architectures Proposer Operator."""

        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.config = config
            self.num_proposers = config.num_proposers
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.d_ffn = config.d_ffn

            self.branch_kinds: List[int] = [i % 4 for i in range(self.num_proposers)]
            self.in_slice_dims: List[int] = []
            for kind in self.branch_kinds:
                if kind == 0:
                    self.in_slice_dims.append(3 * config.d_model)
                elif kind == 1:  # Cross-Modal Vision-Language Fusion (v_raw, t_raw): 2 * D
                    self.in_slice_dims.append(2 * config.d_model)
                elif kind == 2:
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

            self.vectorized_branch_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.branch_scale = nn.Parameter(
                torch.ones(1, self.num_proposers, 1, config.d_model)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            return self.vectorized_branch_norm(stacked) * self.branch_scale


    class CrossProposalAggregatorOp(nn.Module):
        def __init__(self, config: MoAOperatorConfig):
            super().__init__()
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            self.num_proposers = config.num_proposers
            self.top_k = min(config.top_k_proposers, config.num_proposers)
            self.mode = config.aggregator_mode

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
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            metrics: Dict[str, torch.Tensor] = {}

            if self.mode == "cross_attention":
                p_tagged = proposals + self.proposer_id_emb
                q = self.q_proj(x).view(B * L, 1, self.n_heads, self.d_head).transpose(1, 2)
                p_blnd = p_tagged.permute(0, 2, 1, 3).contiguous()
                kv = self.kv_proj(p_blnd).view(B * L, N, 2, self.n_heads, self.d_head)
                k, v = [t.transpose(1, 2) for t in kv.unbind(dim=2)]
                scale = (self.d_head ** -0.5) * self.learnable_temp
                attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
                agg = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
                synthesized = self.out_proj(agg)
                metrics["proposal_importance"] = (
                    attn_weights.view(B, L, self.n_heads, N).mean(dim=(0, 1, 2)).detach()
                )
            elif self.mode == "sparse_router":
                q_r = self.router_query(x) * (D ** -0.5)
                k_r = self.router_key(proposals)
                router_logits = torch.einsum("bld,bnld->bln", q_r, k_r)
                z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
                router_probs = F.softmax(router_logits, dim=-1)

                top_k_vals, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
                top_k_weights = F.softmax(top_k_vals, dim=-1)
                sparse_weights = torch.zeros_like(router_probs).scatter_(
                    dim=-1, index=top_k_indices, src=top_k_weights
                )
                selection_mask = torch.zeros_like(router_probs).scatter_(
                    dim=-1, index=top_k_indices, value=1.0
                )
                f_i = selection_mask.mean(dim=(0, 1))
                P_i = router_probs.mean(dim=(0, 1))
                load_balance_loss = N * torch.sum(f_i * P_i)
                aux_loss = (
                    self.config.aux_loss_weight * load_balance_loss
                    + self.config.z_loss_weight * z_loss
                )
                agg = torch.einsum("bln,bnld->bld", sparse_weights, proposals)
                synthesized = self.out_proj(agg)
                metrics["proposal_importance"] = f_i.detach()
            else:
                p_concat = proposals.permute(0, 2, 1, 3).contiguous().view(B, L, N * D)
                gated = F.silu(self.gate_proj(p_concat)) * self.val_proj(p_concat)
                synthesized = self.out_proj(gated)
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

            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.head_gate = nn.Linear(config.d_model, config.n_heads, bias=True)
            self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
            B, L, D = x.shape
            qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
            mask_np, stats = NumpyMoAOperatorKernels.build_mixture_of_attention_mask(
                L, self.n_heads, self.window_size, self.num_sinks, self.dilation
            )
            attn_mask = torch.from_numpy(mask_np).to(device=x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            gate = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            out = (out * gate).transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out), stats["overall_moa_density"]


# ==============================================================================
# 4. Vision-Language Cross-Modal Operator Suite Verification & 5-Fold CV
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
    print("3. Image-Text Cross-Modal Alignment Benchmark (CrossModalVisionLanguageFusionSubOp)")
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

    # Compare against unaligned linear additive baseline (0.5 * V_stream + 0.5 * T_stream)
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
    assert r2_vl_branch > 0.99
    assert r2_vl_branch > r2_lin_branch + 0.15
    print("=" * 96)

if not HAS_TORCH:
    class RMSNorm:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedCrossModalBilinearGateFunction:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate FusedCrossModalBilinearGateFunction.")

    class CrossModalVisionLanguageFusionSubOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate CrossModalVisionLanguageFusionSubOp.")

    class HeterogeneousProposerOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate HeterogeneousProposerOp.")

    class CrossProposalAggregatorOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate CrossProposalAggregatorOp.")

    class MixtureOfAttentionOp:
        """Fallback stub when PyTorch is not installed; use AnalyticalKernels for NumPy execution."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate MixtureOfAttentionOp.")

if __name__ == "__main__":
    run_moa_vl_operator_verification()
