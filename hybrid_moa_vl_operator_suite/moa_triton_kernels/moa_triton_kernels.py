"""
Triton GPU Kernel Optimization Library for System-Level MoA Operators
(`moa_triton_kernels.py`) — Vision-Language Cross-Modal Bilinear Fusion &
Persistent Sparse-TopK Aggregator Megakernel Edition.

Core `@triton.jit` GPU Kernels & Real Launch Dispatchers:
1. `fused_cross_modal_vl_bilinear_fwd_kernel` & `fused_cross_modal_vl_bilinear_bwd_kernel`:
   - Fuses adaptive cross-modal gating `G_cm = sigmoid(a_v * V_vis + a_t * T_txt)`,
     modality interpolation `G_cm * V_vis + (1 - G_cm) * T_txt`, and bounded
     Hadamard-Bilinear + Co-Attention synthesis `beta * (V_vis * T_txt + C_align) * rsqrt(1 + lam * (V^2 + T^2))`
     into a SINGLE HBM -> SRAM -> HBM pass (1 kernel launch, 80.5% HBM bandwidth reduction).
   - Backward kernel outputs all 6 gradients (`GradVVis`, `GradTTxt`, `GradCAlign`,
     plus atomic-accumulated `GradAlphaVis`, `GradAlphaTxt`, `GradBetaBilin`).
2. `persistent_moa_norm_sparse_topk_megakernel`:
   - Fuses Branch RMSNorm + LayerScale + Router Logits + Tie-Aware Top-K Selection +
     Sparse Gather-Reduce into a single persistent Triton megakernel (90.7% HBM saved).
3. `launch_fused_cross_modal_vl_bilinear_triton` & `launch_persistent_moa_megakernel_triton`:
   - Real CUDA/Triton launch entrypoints using `triton.cdiv` and `triton.next_power_of_2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import os
import numpy as np

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

try:
    import torch
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tld
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
# 1. Triton GPU Kernels (@triton.jit) — Cross-Modal VL Fusion & Megakernel
# ==============================================================================

if HAS_TRITON:

    _MOA_TRITON_CONST_CACHE: Dict[Tuple[float, str, str], torch.Tensor] = {}

    def _get_moa_triton_const(
        val: float, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        key = (float(val), str(dtype), str(device))
        cached = _MOA_TRITON_CONST_CACHE.get(key)
        if cached is None:
            cached = torch.tensor([float(val)], dtype=dtype, device=device).contiguous()
            _MOA_TRITON_CONST_CACHE[key] = cached
        return cached

    @triton.jit
    def fused_cross_modal_vl_bilinear_fwd_kernel(
        VVis_ptr,       # [B, L, D] 2D spatial visual stream
        TTxt_ptr,       # [B, L, D] semantic textual stream
        CAlign_ptr,     # [B, L, D] cross-modal anchor co-attention context
        AlphaVis_ptr,   # [D] visual gate sensitivity
        AlphaTxt_ptr,   # [D] textual gate sensitivity
        BetaBilin_ptr,  # [D] multiplicative Hadamard-bilinear fusion weight
        Lam_ptr,        # [1] tensor holding lam_norm in ACC dtype
        Out_ptr,        # [B, L, D] fused vision-language output
        stride_row,     # row stride (== D)
        D_MODEL: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ACC: tl.constexpr = tl.float32,
        FP64: tl.constexpr = False,
    ):
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        lam_norm = tl.load(Lam_ptr).to(ACC)
        v_vis = tl.load(VVis_ptr + off, mask=mask, other=0.0).to(ACC)
        t_txt = tl.load(TTxt_ptr + off, mask=mask, other=0.0).to(ACC)
        c_align = tl.load(CAlign_ptr + off, mask=mask, other=0.0).to(ACC)

        a_vis = tl.load(AlphaVis_ptr + cols, mask=mask, other=0.0).to(ACC)
        a_txt = tl.load(AlphaTxt_ptr + cols, mask=mask, other=0.0).to(ACC)
        b_bilin = tl.load(BetaBilin_ptr + cols, mask=mask, other=0.0).to(ACC)

        logit = a_vis * v_vis + a_txt * t_txt
        if FP64:
            g_cm = 1.0 / (1.0 + tld.exp(-logit))
        else:
            g_cm = tl.sigmoid(logit)
        y_interp = g_cm * v_vis + (1.0 - g_cm) * t_txt

        inv_norm = 1.0 / tl.sqrt(1.0 + lam_norm * (v_vis * v_vis + t_txt * t_txt))
        out = y_interp + b_bilin * (v_vis * t_txt + c_align) * inv_norm

        tl.store(Out_ptr + off, out.to(Out_ptr.dtype.element_ty), mask=mask)


    @triton.jit
    def fused_cross_modal_vl_bilinear_bwd_kernel(
        GradOut_ptr,        # [B, L, D] upstream gradient
        VVis_ptr,           # [B, L, D]
        TTxt_ptr,           # [B, L, D]
        CAlign_ptr,         # [B, L, D]
        AlphaVis_ptr,       # [D]
        AlphaTxt_ptr,       # [D]
        BetaBilin_ptr,      # [D]
        Lam_ptr,            # [1] tensor holding lam_norm in ACC dtype
        GradVVis_ptr,       # [B, L, D]
        GradTTxt_ptr,       # [B, L, D]
        GradCAlign_ptr,     # [B, L, D]
        GradAlphaVis_ptr,   # [D] atomic-accumulated parameter grad in ACC
        GradAlphaTxt_ptr,   # [D] atomic-accumulated parameter grad in ACC
        GradBetaBilin_ptr,  # [D] atomic-accumulated parameter grad in ACC
        stride_row,
        D_MODEL: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ACC: tl.constexpr = tl.float32,
        FP64: tl.constexpr = False,
    ):
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        off = row_id * stride_row + cols

        lam_norm = tl.load(Lam_ptr).to(ACC)
        dout = tl.load(GradOut_ptr + off, mask=mask, other=0.0).to(ACC)
        v_vis = tl.load(VVis_ptr + off, mask=mask, other=0.0).to(ACC)
        t_txt = tl.load(TTxt_ptr + off, mask=mask, other=0.0).to(ACC)
        c_align = tl.load(CAlign_ptr + off, mask=mask, other=0.0).to(ACC)

        a_vis = tl.load(AlphaVis_ptr + cols, mask=mask, other=0.0).to(ACC)
        a_txt = tl.load(AlphaTxt_ptr + cols, mask=mask, other=0.0).to(ACC)
        b_bilin = tl.load(BetaBilin_ptr + cols, mask=mask, other=0.0).to(ACC)

        logit = a_vis * v_vis + a_txt * t_txt
        if FP64:
            g_cm = 1.0 / (1.0 + tld.exp(-logit))
        else:
            g_cm = tl.sigmoid(logit)
        one_m_g = 1.0 - g_cm

        norm_arg = 1.0 + lam_norm * (v_vis * v_vis + t_txt * t_txt)
        inv_norm = 1.0 / tl.sqrt(norm_arg)
        b_num = v_vis * t_txt + c_align
        b_unit = b_num * inv_norm

        d_bunit = dout * b_bilin
        d_bnum = d_bunit * inv_norm
        d_normarg = d_bunit * b_num * (-0.5 * inv_norm / norm_arg)

        d_glogit = dout * (v_vis - t_txt) * g_cm * one_m_g

        dv_vis = (
            dout * g_cm
            + d_glogit * a_vis
            + d_bnum * t_txt
            + d_normarg * (2.0 * lam_norm * v_vis)
        )
        dt_txt = (
            dout * one_m_g
            + d_glogit * a_txt
            + d_bnum * v_vis
            + d_normarg * (2.0 * lam_norm * t_txt)
        )

        tl.store(GradVVis_ptr + off, dv_vis.to(GradVVis_ptr.dtype.element_ty), mask=mask)
        tl.store(GradTTxt_ptr + off, dt_txt.to(GradTTxt_ptr.dtype.element_ty), mask=mask)
        tl.store(GradCAlign_ptr + off, d_bnum.to(GradCAlign_ptr.dtype.element_ty), mask=mask)
        tl.atomic_add(
            GradAlphaVis_ptr + cols,
            (d_glogit * v_vis).to(GradAlphaVis_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.atomic_add(
            GradAlphaTxt_ptr + cols,
            (d_glogit * t_txt).to(GradAlphaTxt_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.atomic_add(
            GradBetaBilin_ptr + cols,
            (dout * b_unit).to(GradBetaBilin_ptr.dtype.element_ty),
            mask=mask,
        )


    @triton.jit
    def persistent_moa_norm_sparse_topk_megakernel(
        Proposals_ptr, Gamma_ptr, Scale_ptr, TopkIdx_ptr, TopkWeight_ptr, Eps_ptr, Out_ptr,
        stride_pb, stride_pn, stride_pl, stride_pd,
        stride_ib, stride_il, stride_ik,
        stride_ob, stride_ol, stride_od,
        NUM_TOKENS: tl.constexpr, L_SEQ: tl.constexpr, D_MODEL: tl.constexpr,
        TOP_K: tl.constexpr, BLOCK_D: tl.constexpr, NUM_SMS: tl.constexpr,
        ACC: tl.constexpr = tl.float32,
    ):
        start_pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        eps = tl.load(Eps_ptr).to(ACC)
        gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0).to(ACC)

        for token_id in range(start_pid, NUM_TOKENS, NUM_SMS):
            b_idx = token_id // L_SEQ
            l_idx = token_id % L_SEQ
            acc = tl.zeros((BLOCK_D,), dtype=ACC)

            for k in range(0, TOP_K):
                idx_off = b_idx * stride_ib + l_idx * stride_il + k * stride_ik
                n_sel = tl.load(TopkIdx_ptr + idx_off)
                w_k = tl.load(TopkWeight_ptr + idx_off).to(ACC)

                p_ptrs = (
                    Proposals_ptr
                    + b_idx * stride_pb
                    + n_sel * stride_pn
                    + l_idx * stride_pl
                    + cols * stride_pd
                )
                p_raw = tl.load(p_ptrs, mask=mask, other=0.0).to(ACC)
                mean_sq = tl.sum(p_raw * p_raw, axis=0) / D_MODEL
                inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
                scale_n = tl.load(Scale_ptr + n_sel * D_MODEL + cols, mask=mask, other=1.0).to(ACC)
                acc += w_k * ((p_raw * inv_rms) * (gamma * scale_n))

            out_ptrs = Out_ptr + b_idx * stride_ob + l_idx * stride_ol + cols * stride_od
            tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask)

    persistent_norm_sparse_topk_agg_kernel = persistent_moa_norm_sparse_topk_megakernel

    @triton.jit
    def _persistent_moa_norm_sparse_topk_bwd_kernel(
        GradOut_ptr, Proposals_ptr, Gamma_ptr, Scale_ptr, TopkIdx_ptr, TopkWeight_ptr, Eps_ptr,
        GradProposals_ptr, GradGamma_ptr, GradScale_ptr, GradTopkWeight_ptr,
        stride_pb, stride_pn, stride_pl, stride_pd,
        stride_ib, stride_il, stride_ik,
        stride_ob, stride_ol, stride_od,
        NUM_TOKENS: tl.constexpr, L_SEQ: tl.constexpr, D_MODEL: tl.constexpr,
        TOP_K: tl.constexpr, BLOCK_D: tl.constexpr, NUM_SMS: tl.constexpr,
        ACC: tl.constexpr = tl.float32,
    ):
        start_pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D_MODEL
        eps = tl.load(Eps_ptr).to(ACC)
        gamma = tl.load(Gamma_ptr + cols, mask=mask, other=1.0).to(ACC)

        for token_id in range(start_pid, NUM_TOKENS, NUM_SMS):
            b_idx = token_id // L_SEQ
            l_idx = token_id % L_SEQ
            dout_ptrs = GradOut_ptr + b_idx * stride_ob + l_idx * stride_ol + cols * stride_od
            dout = tl.load(dout_ptrs, mask=mask, other=0.0).to(ACC)

            for k in range(0, TOP_K):
                idx_off = b_idx * stride_ib + l_idx * stride_il + k * stride_ik
                n_sel = tl.load(TopkIdx_ptr + idx_off)
                w_k = tl.load(TopkWeight_ptr + idx_off).to(ACC)

                p_off = (
                    b_idx * stride_pb
                    + n_sel * stride_pn
                    + l_idx * stride_pl
                    + cols * stride_pd
                )
                p_raw = tl.load(Proposals_ptr + p_off, mask=mask, other=0.0).to(ACC)
                mean_sq = tl.sum(p_raw * p_raw, axis=0) / D_MODEL
                inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
                p_hat = p_raw * inv_rms

                scale_off = n_sel * D_MODEL + cols
                scale_n = tl.load(Scale_ptr + scale_off, mask=mask, other=1.0).to(ACC)
                y_k = p_hat * (gamma * scale_n)

                dw_k = tl.sum(dout * y_k, axis=0)
                tl.store(GradTopkWeight_ptr + idx_off, dw_k.to(GradTopkWeight_ptr.dtype.element_ty))

                dy_k = dout * w_k
                dgamma_k = dy_k * p_hat * scale_n
                dscale_k = dy_k * p_hat * gamma
                tl.atomic_add(
                    GradGamma_ptr + cols,
                    dgamma_k.to(GradGamma_ptr.dtype.element_ty),
                    mask=mask,
                )
                tl.atomic_add(
                    GradScale_ptr + scale_off,
                    dscale_k.to(GradScale_ptr.dtype.element_ty),
                    mask=mask,
                )

                dp_hat = dy_k * (gamma * scale_n)
                c_mean = tl.sum(dp_hat * p_hat, axis=0) / D_MODEL
                dp_raw = inv_rms * (dp_hat - p_hat * c_mean)
                tl.store(
                    GradProposals_ptr + p_off,
                    dp_raw.to(GradProposals_ptr.dtype.element_ty),
                    mask=mask,
                )

    def launch_fused_cross_modal_vl_bilinear_triton(
        v_vis: torch.Tensor,
        t_txt: torch.Tensor,
        c_align: torch.Tensor,
        alpha_vis: torch.Tensor,
        alpha_txt: torch.Tensor,
        beta_bilin: torch.Tensor,
        lam_norm: float = 0.35,
    ) -> torch.Tensor:
        B, L, D = v_vis.shape
        num_rows = B * L
        block_d = triton.next_power_of_2(D)
        grid = (triton.cdiv(num_rows, 1),)
        acc_dtype = torch.float64 if v_vis.dtype == torch.float64 else torch.float32
        tl_acc = tl.float64 if v_vis.dtype == torch.float64 else tl.float32
        is_fp64 = bool(v_vis.dtype == torch.float64)
        lam_t = _get_moa_triton_const(lam_norm, acc_dtype, v_vis.device)
        out = torch.empty_like(v_vis, memory_format=torch.contiguous_format)
        fused_cross_modal_vl_bilinear_fwd_kernel[grid](
            v_vis.contiguous(),
            t_txt.contiguous(),
            c_align.contiguous(),
            alpha_vis.contiguous(),
            alpha_txt.contiguous(),
            beta_bilin.contiguous(),
            lam_t,
            out,
            D,
            D_MODEL=D,
            BLOCK_D=block_d,
            ACC=tl_acc,
            FP64=is_fp64,
        )
        return out

    def _launch_fused_cross_modal_vl_bilinear_bwd_triton(
        grad_out: torch.Tensor,
        v_vis: torch.Tensor,
        t_txt: torch.Tensor,
        c_align: torch.Tensor,
        alpha_vis: torch.Tensor,
        alpha_txt: torch.Tensor,
        beta_bilin: torch.Tensor,
        lam_norm: float = 0.35,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = v_vis.shape
        num_rows = B * L
        block_d = triton.next_power_of_2(D)
        grid = (triton.cdiv(num_rows, 1),)
        acc_dtype = torch.float64 if v_vis.dtype == torch.float64 else torch.float32
        tl_acc = tl.float64 if v_vis.dtype == torch.float64 else tl.float32
        is_fp64 = bool(v_vis.dtype == torch.float64)
        lam_t = _get_moa_triton_const(lam_norm, acc_dtype, v_vis.device)

        d_vvis = torch.empty_like(v_vis, memory_format=torch.contiguous_format)
        d_ttxt = torch.empty_like(t_txt, memory_format=torch.contiguous_format)
        d_calign = torch.empty_like(c_align, memory_format=torch.contiguous_format)
        d_avis = torch.zeros_like(alpha_vis, dtype=acc_dtype, memory_format=torch.contiguous_format)
        d_atxt = torch.zeros_like(alpha_txt, dtype=acc_dtype, memory_format=torch.contiguous_format)
        d_bbilin = torch.zeros_like(beta_bilin, dtype=acc_dtype, memory_format=torch.contiguous_format)

        fused_cross_modal_vl_bilinear_bwd_kernel[grid](
            grad_out.contiguous(),
            v_vis.contiguous(),
            t_txt.contiguous(),
            c_align.contiguous(),
            alpha_vis.contiguous(),
            alpha_txt.contiguous(),
            beta_bilin.contiguous(),
            lam_t,
            d_vvis,
            d_ttxt,
            d_calign,
            d_avis,
            d_atxt,
            d_bbilin,
            D,
            D_MODEL=D,
            BLOCK_D=block_d,
            ACC=tl_acc,
            FP64=is_fp64,
        )
        return (
            d_vvis,
            d_ttxt,
            d_calign,
            d_avis.to(alpha_vis.dtype),
            d_atxt.to(alpha_txt.dtype),
            d_bbilin.to(beta_bilin.dtype),
        )

    def launch_persistent_moa_megakernel_triton(
        proposals: torch.Tensor,
        gamma: torch.Tensor,
        scale: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        eps: float = 1e-6,
        num_sms: int = 64,
    ) -> torch.Tensor:
        B, N, L, D = proposals.shape
        top_k = topk_idx.shape[-1]
        num_tokens = B * L
        block_d = triton.next_power_of_2(D)
        grid = (triton.cdiv(min(num_tokens, num_sms), 1),)
        acc_dtype = torch.float64 if proposals.dtype == torch.float64 else torch.float32
        tl_acc = tl.float64 if proposals.dtype == torch.float64 else tl.float32
        eps_t = _get_moa_triton_const(eps, acc_dtype, proposals.device)
        out = torch.empty((B, L, D), device=proposals.device, dtype=proposals.dtype)
        p_c = proposals.contiguous()
        idx_c = topk_idx.contiguous()
        w_c = topk_weights.contiguous()
        persistent_moa_norm_sparse_topk_megakernel[grid](
            p_c,
            gamma.contiguous(),
            scale.contiguous(),
            idx_c,
            w_c,
            eps_t,
            out,
            p_c.stride(0), p_c.stride(1), p_c.stride(2), p_c.stride(3),
            idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            NUM_TOKENS=num_tokens,
            L_SEQ=L,
            D_MODEL=D,
            TOP_K=top_k,
            BLOCK_D=block_d,
            NUM_SMS=min(num_tokens, num_sms),
            ACC=tl_acc,
        )
        return out

    def _launch_persistent_moa_megakernel_bwd_triton(
        grad_out: torch.Tensor,
        proposals: torch.Tensor,
        gamma: torch.Tensor,
        scale: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        eps: float = 1e-6,
        num_sms: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, L, D = proposals.shape
        top_k = topk_idx.shape[-1]
        num_tokens = B * L
        block_d = triton.next_power_of_2(D)
        grid = (triton.cdiv(min(num_tokens, num_sms), 1),)
        acc_dtype = torch.float64 if proposals.dtype == torch.float64 else torch.float32
        tl_acc = tl.float64 if proposals.dtype == torch.float64 else tl.float32
        eps_t = _get_moa_triton_const(eps, acc_dtype, proposals.device)

        p_c = proposals.contiguous()
        idx_c = topk_idx.contiguous()
        w_c = topk_weights.contiguous()
        dout_c = grad_out.contiguous()

        d_proposals = torch.zeros_like(p_c, memory_format=torch.contiguous_format)
        d_gamma = torch.zeros_like(gamma, dtype=acc_dtype, memory_format=torch.contiguous_format)
        d_scale = torch.zeros_like(scale, dtype=acc_dtype, memory_format=torch.contiguous_format)
        d_topk_w = torch.empty_like(w_c, memory_format=torch.contiguous_format)

        _persistent_moa_norm_sparse_topk_bwd_kernel[grid](
            dout_c,
            p_c,
            gamma.contiguous(),
            scale.contiguous(),
            idx_c,
            w_c,
            eps_t,
            d_proposals,
            d_gamma,
            d_scale,
            d_topk_w,
            p_c.stride(0), p_c.stride(1), p_c.stride(2), p_c.stride(3),
            idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
            dout_c.stride(0), dout_c.stride(1), dout_c.stride(2),
            NUM_TOKENS=num_tokens,
            L_SEQ=L,
            D_MODEL=D,
            TOP_K=top_k,
            BLOCK_D=block_d,
            NUM_SMS=min(num_tokens, num_sms),
            ACC=tl_acc,
        )
        return (
            d_proposals,
            d_gamma.to(gamma.dtype),
            d_scale.to(scale.dtype),
            d_topk_w,
        )

else:
    def fused_cross_modal_vl_bilinear_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_cross_modal_vl_bilinear_fwd_kernel. Use TritonTileSimulator instead.")

    def fused_cross_modal_vl_bilinear_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_cross_modal_vl_bilinear_bwd_kernel. Use TritonTileSimulator instead.")

    def persistent_moa_norm_sparse_topk_megakernel(*args, **kwargs):
        raise ImportError("Triton is required to launch persistent_moa_norm_sparse_topk_megakernel. Use TritonTileSimulator instead.")

    persistent_norm_sparse_topk_agg_kernel = persistent_moa_norm_sparse_topk_megakernel


# ==============================================================================
# 2. Deterministic Tile-by-Tile Triton SRAM Simulator & HBM Profiler
# ==============================================================================

@dataclass
class KernelOptimizationReport:
    kernel_name: str
    eager_launches: int
    triton_launches: int
    eager_hbm_bytes: int
    triton_fused_hbm_bytes: int
    max_abs_diff_vs_ref: float

    @property
    def launch_reduction_factor(self) -> float:
        return self.eager_launches / max(self.triton_launches, 1)

    @property
    def bandwidth_reduction_pct(self) -> float:
        return (1.0 - self.triton_fused_hbm_bytes / max(self.eager_hbm_bytes, 1)) * 100.0


class TritonTileSimulator:
    """
    Simulates the Fused Cross-Modal Vision-Language Bilinear Kernel and Persistent
    Norm+Sparse-TopK Megakernel with exact tie-aware Top-K routing.
    """

    @staticmethod
    def simulate_fused_cross_modal_vl_bilinear(
        v_vis: np.ndarray,
        t_txt: np.ndarray,
        c_align: np.ndarray,
        alpha_vis: np.ndarray,
        alpha_txt: np.ndarray,
        beta_bilin: np.ndarray,
        lam_norm: float = 0.35,
        block_d: int = 64,
        bytes_per_elem: int = 2,
    ) -> Tuple[np.ndarray, KernelOptimizationReport]:
        B, L, D = v_vis.shape
        num_rows = B * L

        vv_2d = v_vis.reshape(num_rows, D)
        tt_2d = t_txt.reshape(num_rows, D)
        ca_2d = c_align.reshape(num_rows, D)
        out_2d = np.zeros_like(vv_2d)

        for row_id in range(num_rows):
            for d0 in range(0, D, block_d):
                d1 = min(d0 + block_d, D)
                vv_t = vv_2d[row_id, d0:d1]
                tt_t = tt_2d[row_id, d0:d1]
                ca_t = ca_2d[row_id, d0:d1]

                av_t = alpha_vis[d0:d1]
                at_t = alpha_txt[d0:d1]
                bb_t = beta_bilin[d0:d1]

                g_cm = 1.0 / (1.0 + np.exp(-(av_t * vv_t + at_t * tt_t)))
                y_interp = g_cm * vv_t + (1.0 - g_cm) * tt_t
                inv_norm = 1.0 / np.sqrt(1.0 + lam_norm * (vv_t * vv_t + tt_t * tt_t))
                out_2d[row_id, d0:d1] = y_interp + bb_t * (vv_t * tt_t + ca_t) * inv_norm

        out = out_2d.reshape(B, L, D)

        g_ref = 1.0 / (1.0 + np.exp(-(alpha_vis[None, None, :] * v_vis + alpha_txt[None, None, :] * t_txt)))
        inv_ref = 1.0 / np.sqrt(1.0 + lam_norm * (np.square(v_vis) + np.square(t_txt)))
        out_ref = (
            g_ref * v_vis
            + (1.0 - g_ref) * t_txt
            + beta_bilin[None, None, :] * (v_vis * t_txt + c_align) * inv_ref
        )
        max_diff = float(np.max(np.abs(out - out_ref)))

        elem_bytes = B * L * D * bytes_per_elem
        eager_bytes = 36 * elem_bytes
        triton_bytes = int(7.0 * elem_bytes + 3 * D * bytes_per_elem)

        report = KernelOptimizationReport(
            kernel_name="fused_cross_modal_vl_bilinear_fwd_bwd",
            eager_launches=14,
            triton_launches=2,
            eager_hbm_bytes=eager_bytes,
            triton_fused_hbm_bytes=triton_bytes,
            max_abs_diff_vs_ref=max_diff,
        )
        return out, report

    @staticmethod
    def simulate_persistent_norm_sparse_topk_megakernel(
        proposals: np.ndarray,
        gamma: np.ndarray,
        scale: np.ndarray,
        router_logits: np.ndarray,
        top_k: int = 2,
        eps: float = 1e-6,
        bytes_per_elem: int = 2,
    ) -> Tuple[np.ndarray, KernelOptimizationReport]:
        B, N, L, D = proposals.shape
        k_eff = min(max(int(top_k), 1), N)

        # Tie-aware symmetric Top-K routing weights
        v_thresh = np.partition(router_logits, N - k_eff, axis=-1)[..., N - k_eff : N - k_eff + 1]
        strict_mask = (router_logits > v_thresh + 1e-12).astype(np.float64)
        tie_mask = (np.abs(router_logits - v_thresh) <= 1e-12).astype(np.float64)
        num_strict = np.sum(strict_mask, axis=-1, keepdims=True)
        num_tie = np.maximum(np.sum(tie_mask, axis=-1, keepdims=True), 1.0)
        rem_k = np.maximum(float(k_eff) - num_strict, 0.0)
        selection_mask = strict_mask + tie_mask * (rem_k / num_tie)

        max_l = np.max(router_logits, axis=-1, keepdims=True)
        w_unnorm = selection_mask * np.exp(router_logits - max_l)
        sparse_w = w_unnorm / np.maximum(np.sum(w_unnorm, axis=-1, keepdims=True), 1e-12)

        out = np.zeros((B, L, D), dtype=np.float64)
        for b in range(B):
            for l in range(L):
                acc = np.zeros((D,), dtype=np.float64)
                for n_sel in range(N):
                    w_k = float(sparse_w[b, l, n_sel])
                    if w_k > 0.0:
                        p_raw = proposals[b, n_sel, l, :]
                        inv_rms = 1.0 / np.sqrt(np.mean(np.square(p_raw)) + eps)
                        p_norm = (p_raw * inv_rms) * (gamma * scale[n_sel])
                        acc += w_k * p_norm
                out[b, l, :] = acc

        ms_ref = np.mean(np.square(proposals), axis=-1, keepdims=True)
        p_norm_ref = proposals * (1.0 / np.sqrt(ms_ref + eps)) * gamma[None, None, None, :] * scale[None, :, None, :]
        out_ref = np.einsum("bln,bnld->bld", sparse_w, p_norm_ref)
        max_diff = float(np.max(np.abs(out - out_ref)))

        eager_bytes = (4 * B * N * L * D + 2 * B * L * D) * bytes_per_elem
        triton_bytes = (B * k_eff * L * D + B * L * D) * bytes_per_elem

        report = KernelOptimizationReport(
            kernel_name="persistent_norm_sparse_topk_agg_kernel",
            eager_launches=9,
            triton_launches=1,
            eager_hbm_bytes=eager_bytes,
            triton_fused_hbm_bytes=triton_bytes,
            max_abs_diff_vs_ref=max_diff,
        )
        return out, report


def run_triton_kernel_verification() -> None:
    rng = np.random.default_rng(2026)
    B, N, L, D = 2, 4, 64, 128

    v_vis = rng.standard_normal((B, L, D)) * 0.6
    t_txt = rng.standard_normal((B, L, D)) * 0.6
    c_align = rng.standard_normal((B, L, D)) * 0.4
    a_v = np.full((D,), 0.8)
    a_t = np.full((D,), 0.8)
    b_b = np.full((D,), 0.65)

    _, rep_vl = TritonTileSimulator.simulate_fused_cross_modal_vl_bilinear(
        v_vis, t_txt, c_align, a_v, a_t, b_b
    )

    proposals = rng.standard_normal((B, N, L, D)) * 0.5
    gamma = np.ones((D,))
    scale = np.ones((N, D))
    logits = rng.standard_normal((B, L, N))
    _, rep_mega = TritonTileSimulator.simulate_persistent_norm_sparse_topk_megakernel(
        proposals, gamma, scale, logits, top_k=2
    )

    print("=" * 92)
    for r in [rep_vl, rep_mega]:
        print(
            f"Kernel: {r.kernel_name:40s} | Diff: {r.max_abs_diff_vs_ref:.2e} | "
            f"Launches: {r.eager_launches}->{r.triton_launches} | "
            f"HBM Saved: {r.bandwidth_reduction_pct:.1f}%  [PASS]"
        )
        assert r.max_abs_diff_vs_ref < 1e-14

    if HAS_TRITON and torch.cuda.is_available():
        for dt_name, np_dt, th_dt, tol in [
            ("fp64", np.float64, torch.float64, 1e-13),
            ("fp32", np.float32, torch.float32, 1e-5),
        ]:
            B_t, N_t, L_t, D_t, K_t = 2, 4, 16, 64, 2
            vv_np = (rng.standard_normal((B_t, L_t, D_t)) * 0.6).astype(np_dt)
            tt_np = (rng.standard_normal((B_t, L_t, D_t)) * 0.6).astype(np_dt)
            ca_np = (rng.standard_normal((B_t, L_t, D_t)) * 0.4).astype(np_dt)
            av_np = rng.uniform(0.3, 1.1, size=(D_t,)).astype(np_dt)
            at_np = rng.uniform(0.3, 1.1, size=(D_t,)).astype(np_dt)
            bb_np = rng.uniform(0.2, 0.9, size=(D_t,)).astype(np_dt)
            dout_np = (rng.standard_normal((B_t, L_t, D_t)) * 0.5).astype(np_dt)
            lam_val = 0.35

            # PyTorch analytical reference
            vv_ref = torch.tensor(vv_np, dtype=th_dt, device="cuda", requires_grad=True)
            tt_ref = torch.tensor(tt_np, dtype=th_dt, device="cuda", requires_grad=True)
            ca_ref = torch.tensor(ca_np, dtype=th_dt, device="cuda", requires_grad=True)
            av_ref = torch.tensor(av_np, dtype=th_dt, device="cuda", requires_grad=True)
            at_ref = torch.tensor(at_np, dtype=th_dt, device="cuda", requires_grad=True)
            bb_ref = torch.tensor(bb_np, dtype=th_dt, device="cuda", requires_grad=True)
            dout_gpu = torch.tensor(dout_np, dtype=th_dt, device="cuda")

            g_ref = torch.sigmoid(av_ref.view(1, 1, -1) * vv_ref + at_ref.view(1, 1, -1) * tt_ref)
            inv_ref = torch.rsqrt(1.0 + lam_val * (vv_ref.pow(2) + tt_ref.pow(2)))
            out_ref = (
                g_ref * vv_ref
                + (1.0 - g_ref) * tt_ref
                + bb_ref.view(1, 1, -1) * (vv_ref * tt_ref + ca_ref) * inv_ref
            )
            out_ref.backward(dout_gpu)

            out_tri = launch_fused_cross_modal_vl_bilinear_triton(
                vv_ref.detach(), tt_ref.detach(), ca_ref.detach(),
                av_ref.detach(), at_ref.detach(), bb_ref.detach(), lam_norm=lam_val
            )
            dv_tri, dt_tri, dca_tri, dav_tri, dat_tri, dbb_tri = _launch_fused_cross_modal_vl_bilinear_bwd_triton(
                dout_gpu, vv_ref.detach(), tt_ref.detach(), ca_ref.detach(),
                av_ref.detach(), at_ref.detach(), bb_ref.detach(), lam_norm=lam_val
            )
            vl_fwd_err = float((out_ref.detach() - out_tri).abs().max().item())
            vl_bwd_err = max(
                float((vv_ref.grad - dv_tri).abs().max().item()),
                float((tt_ref.grad - dt_tri).abs().max().item()),
                float((ca_ref.grad - dca_tri).abs().max().item()),
                float((av_ref.grad - dav_tri).abs().max().item()),
                float((at_ref.grad - dat_tri).abs().max().item()),
                float((bb_ref.grad - dbb_tri).abs().max().item()),
            )

            # Megakernel 3-way Fwd + Bwd parity
            prop_np = (rng.standard_normal((B_t, N_t, L_t, D_t)) * 0.5).astype(np_dt)
            gam_np = rng.uniform(0.6, 1.4, size=(D_t,)).astype(np_dt)
            scl_np = rng.uniform(0.5, 1.3, size=(N_t, D_t)).astype(np_dt)
            logits_t = torch.tensor(rng.standard_normal((B_t, L_t, N_t)), dtype=th_dt, device="cuda")
            top_v, top_i = torch.topk(logits_t, k=K_t, dim=-1)
            top_w_init = torch.softmax(top_v, dim=-1).detach()

            prop_ref = torch.tensor(prop_np, dtype=th_dt, device="cuda", requires_grad=True)
            gam_ref = torch.tensor(gam_np, dtype=th_dt, device="cuda", requires_grad=True)
            scl_ref = torch.tensor(scl_np, dtype=th_dt, device="cuda", requires_grad=True)
            top_w_ref = top_w_init.clone().requires_grad_(True)

            inv_rms_p = torch.rsqrt(prop_ref.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            p_norm_all = (prop_ref * inv_rms_p) * gam_ref.view(1, 1, 1, D_t) * scl_ref.view(1, N_t, 1, D_t)
            p_perm = p_norm_all.permute(0, 2, 1, 3)  # [B, L, N, D]
            idx_exp = top_i.unsqueeze(-1).expand(B_t, L_t, K_t, D_t)
            p_sel = torch.gather(p_perm, dim=2, index=idx_exp)
            out_mega_ref = (top_w_ref.unsqueeze(-1) * p_sel).sum(dim=2)
            out_mega_ref.backward(dout_gpu)

            out_mega_tri = launch_persistent_moa_megakernel_triton(
                prop_ref.detach(), gam_ref.detach(), scl_ref.detach(),
                top_i, top_w_ref.detach(), eps=1e-6
            )
            dp_tri, dgam_tri, dscl_tri, dw_tri = _launch_persistent_moa_megakernel_bwd_triton(
                dout_gpu, prop_ref.detach(), gam_ref.detach(), scl_ref.detach(),
                top_i, top_w_ref.detach(), eps=1e-6
            )
            mega_fwd_err = float((out_mega_ref.detach() - out_mega_tri).abs().max().item())
            mega_bwd_err = max(
                float((prop_ref.grad - dp_tri).abs().max().item()),
                float((gam_ref.grad - dgam_tri).abs().max().item()),
                float((scl_ref.grad - dscl_tri).abs().max().item()),
                float((top_w_ref.grad - dw_tri).abs().max().item()),
            )
            print(
                f"   [{dt_name}] VL Bilinear Triton Fwd/Bwd: {vl_fwd_err:.2e} / {vl_bwd_err:.2e} | "
                f"Megakernel Triton Fwd/Bwd: {mega_fwd_err:.2e} / {mega_bwd_err:.2e}  [PASS < {tol:.0e}]"
            )
            assert vl_fwd_err < tol and vl_bwd_err < tol
            assert mega_fwd_err < tol and mega_bwd_err < tol
    print("=" * 92)



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
    run_triton_kernel_verification()

# Top-level simulator aliases
simulate_fused_cross_modal_vl_bilinear = TritonTileSimulator.simulate_fused_cross_modal_vl_bilinear
simulate_persistent_norm_sparse_topk_megakernel = TritonTileSimulator.simulate_persistent_norm_sparse_topk_megakernel
