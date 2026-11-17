"""
Quantum Resource Phasing Neural Operator Library.

A standalone, differentiable feature operator inspired by three quantum-resource
optimization benchmarks:

1. Fixed-point inverse-square-root phasing for pairwise Coulomb interactions.
2. Decoded Quantum Interferometry (DQI), using the reliability 1/R(epsilon).
3. Hamiltonian block encoding, using lambda_eff as a resource-normalization gate.

The operator maps hidden states [B, L, D] -> [B, L, D].  Its quantum-resource
quantities are differentiable neural proxies, NOT measurements of an actual
quantum circuit.  Note that the inverse-square-root factor z = (softplus(a) + delta)^(-1/2)
acts both as a phase shift (phi = p + z) and as a bounded-from-above amplitude
gain (up to delta^(-1/2)) modulated by the DQI reliability gate rho in (0, 1] and
block-encoding gate g_block in (0, 1).  No sibling modules are required.

Layers:
- QuantumResourcePhasingConfig
- QuantumResourcePhasingAnalyticalKernels (NumPy forward/backward for both local and pairwise attention paths)
- FusedQuantumResourcePhasingFunction (PyTorch autograd with singularity-free backward & compact saved tensors)
- Fused Triton forward/backward kernels (column-tiled, fp64/fp32-aware, singularity-free)
- QuantumResourcePhasingOp (nn.Module)
- Tile simulator and standalone numerical verification (including extreme negative syndrome stress tests)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
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


LN2 = math.log(2.0)


# ==============================================================================
# 1. Operator configuration
# ==============================================================================

@dataclass
class QuantumResourcePhasingConfig:
    """Configuration for the quantum-resource-inspired feature operator."""

    d_model: int = 256
    n_heads: int = 8
    distance_delta: float = 1e-3
    pair_distance_delta: float = 1.0
    failure_epsilon_max: float = 0.05  # must be < 1/16 for R(epsilon)
    energy_eps: float = 1e-3
    reliability_eps: float = 1e-8
    rms_norm_eps: float = 1e-6
    mixing_mode: Literal["causal", "full"] = "causal"

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0.0 < self.failure_epsilon_max < 1.0 / 16.0:
            raise ValueError("failure_epsilon_max must lie in (0, 1/16).")
        if self.distance_delta <= 0.0 or self.pair_distance_delta <= 0.0:
            raise ValueError("Distance softening constants must be positive.")
        if self.energy_eps <= 0.0:
            raise ValueError("energy_eps must be positive.")


# ==============================================================================
# 2. Optional fused Triton kernels
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def fused_quantum_resource_phasing_fwd_kernel(
        DistanceRaw_ptr,
        Phase0_ptr,
        Syndrome_ptr,
        GapRaw_ptr,
        NormRaw_ptr,
        OutReal_ptr,
        OutImag_ptr,
        OutReliability_ptr,
        stride_row,
        D_MODEL: tl.constexpr,
        DISTANCE_DELTA: tl.constexpr,
        FAILURE_MAX: tl.constexpr,
        ENERGY_EPS: tl.constexpr,
        LN_2: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_FP64: tl.constexpr,
    ):
        """Column-tiled fused inverse-square-root phasing + DQI + block-encoding gate."""
        row_id = tl.program_id(0)
        compute_dtype = tl.float64 if IS_FP64 else tl.float32

        for d0 in range(0, D_MODEL, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D)
            mask = cols < D_MODEL
            offsets = row_id * stride_row + cols

            a = tl.load(DistanceRaw_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            p = tl.load(Phase0_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            s = tl.load(Syndrome_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            u = tl.load(GapRaw_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            v = tl.load(NormRaw_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)

            sp_a = tl.maximum(a, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a)))
            radius = sp_a + DISTANCE_DELTA
            inv_sqrt = 1.0 / tl.sqrt(radius)
            phase = p + inv_sqrt

            sig_s = 1.0 / (1.0 + tl.exp(-s))
            failure = FAILURE_MAX * sig_s
            sqrt_failure = tl.sqrt(failure)
            reliability_argument = 0.5 + 2.0 * sqrt_failure
            reliability = -tl.log(reliability_argument) / LN_2

            sp_u = tl.maximum(u, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(u)))
            sp_v = tl.maximum(v, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(v)))
            gap = sp_u + ENERGY_EPS
            spectral_norm = gap + sp_v + ENERGY_EPS
            lambda_eff = tl.sqrt(gap * (2.0 * spectral_norm - gap))
            block_gate = 1.0 / tl.sqrt(1.0 + lambda_eff * lambda_eff)

            envelope = inv_sqrt * reliability * block_gate
            out_real = envelope * tl.cos(phase)
            out_imag = envelope * tl.sin(phase)

            tl.store(OutReal_ptr + offsets, out_real, mask=mask)
            tl.store(OutImag_ptr + offsets, out_imag, mask=mask)
            tl.store(OutReliability_ptr + offsets, reliability, mask=mask)


    @triton.jit
    def fused_quantum_resource_phasing_bwd_kernel(
        GradReal_ptr,
        GradImag_ptr,
        GradReliability_ptr,
        DistanceRaw_ptr,
        Phase0_ptr,
        Syndrome_ptr,
        GapRaw_ptr,
        NormRaw_ptr,
        GradDistanceRaw_ptr,
        GradPhase0_ptr,
        GradSyndrome_ptr,
        GradGapRaw_ptr,
        GradNormRaw_ptr,
        stride_row,
        D_MODEL: tl.constexpr,
        DISTANCE_DELTA: tl.constexpr,
        FAILURE_MAX: tl.constexpr,
        ENERGY_EPS: tl.constexpr,
        LN_2: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_FP64: tl.constexpr,
    ):
        """Singularity-free analytical backward for the fused quantum resource proxy."""
        row_id = tl.program_id(0)
        compute_dtype = tl.float64 if IS_FP64 else tl.float32

        for d0 in range(0, D_MODEL, BLOCK_D):
            cols = d0 + tl.arange(0, BLOCK_D)
            mask = cols < D_MODEL
            offsets = row_id * stride_row + cols

            grad_real = tl.load(GradReal_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            grad_imag = tl.load(GradImag_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            grad_rel_ext = tl.load(GradReliability_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            a = tl.load(DistanceRaw_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            phase0 = tl.load(Phase0_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            s = tl.load(Syndrome_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            u = tl.load(GapRaw_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)
            v = tl.load(NormRaw_ptr + offsets, mask=mask, other=0.0).to(compute_dtype)

            sig_a = 1.0 / (1.0 + tl.exp(-a))
            sp_a = tl.maximum(a, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a)))
            radius = sp_a + DISTANCE_DELTA
            inv_sqrt = 1.0 / tl.sqrt(radius)
            phase = phase0 + inv_sqrt
            cos_phase = tl.cos(phase)
            sin_phase = tl.sin(phase)

            sig_s = 1.0 / (1.0 + tl.exp(-s))
            failure = FAILURE_MAX * sig_s
            sqrt_failure = tl.sqrt(failure)
            reliability_argument = 0.5 + 2.0 * sqrt_failure
            reliability = -tl.log(reliability_argument) / LN_2

            sig_u = 1.0 / (1.0 + tl.exp(-u))
            sig_v = 1.0 / (1.0 + tl.exp(-v))
            sp_u = tl.maximum(u, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(u)))
            sp_v = tl.maximum(v, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(v)))
            gap = sp_u + ENERGY_EPS
            spectral_norm = gap + sp_v + ENERGY_EPS
            lambda_eff = tl.sqrt(gap * (2.0 * spectral_norm - gap))
            block_gate = 1.0 / tl.sqrt(1.0 + lambda_eff * lambda_eff)
            envelope = inv_sqrt * reliability * block_gate

            d_envelope = grad_real * cos_phase + grad_imag * sin_phase
            d_phase = envelope * (-grad_real * sin_phase + grad_imag * cos_phase)
            d_inv_sqrt = d_envelope * reliability * block_gate + d_phase
            d_reliability_gate = d_envelope * inv_sqrt

            d_a = d_inv_sqrt * (-0.5 * inv_sqrt / radius) * sig_a
            d_p = d_phase

            # Singularity-free cancelled derivative:
            # d(reliability)/ds = -sqrt(failure) * (1 - sig_s) / (LN2 * (0.5 + 2 * sqrt(failure)))
            d_reliability = d_reliability_gate * block_gate + grad_rel_ext
            d_s = (
                -d_reliability
                * sqrt_failure
                * (1.0 - sig_s)
                / (LN_2 * reliability_argument)
            )

            d_block_gate = d_reliability_gate * reliability
            block_gate_cubed = block_gate * block_gate * block_gate
            d_lambda = d_block_gate * (-lambda_eff * block_gate_cubed)
            d_radicand = d_lambda / (2.0 * lambda_eff)
            d_u = d_radicand * (2.0 * spectral_norm) * sig_u
            d_v = d_radicand * (2.0 * gap) * sig_v

            tl.store(GradDistanceRaw_ptr + offsets, d_a, mask=mask)
            tl.store(GradPhase0_ptr + offsets, d_p, mask=mask)
            tl.store(GradSyndrome_ptr + offsets, d_s, mask=mask)
            tl.store(GradGapRaw_ptr + offsets, d_u, mask=mask)
            tl.store(GradNormRaw_ptr + offsets, d_v, mask=mask)



else:
    def fused_quantum_resource_phasing_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_quantum_resource_phasing_fwd_kernel.")

    def fused_quantum_resource_phasing_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to launch fused_quantum_resource_phasing_bwd_kernel.")


# ==============================================================================
# 3. NumPy analytical kernels
# ==============================================================================

class QuantumResourcePhasingAnalyticalKernels:
    """Analytical reference forward/backward and pairwise sequence mixing."""

    @staticmethod
    def softplus(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.result_type(x, np.float64))
        positive = x >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        return out

    @staticmethod
    def dqi_reliability(
        syndrome: np.ndarray,
        failure_epsilon_max: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return reliability=1/R(epsilon), failure proxy, and sqrt(failure)."""
        failure = failure_epsilon_max * QuantumResourcePhasingAnalyticalKernels.sigmoid(syndrome)
        sqrt_failure = np.sqrt(failure)
        argument = 0.5 + 2.0 * sqrt_failure
        reliability = -np.log(argument) / LN2
        return reliability, failure, sqrt_failure

    @staticmethod
    def fused_resource_phasing_forward(
        distance_raw: np.ndarray,
        phase0: np.ndarray,
        syndrome: np.ndarray,
        gap_raw: np.ndarray,
        norm_raw: np.ndarray,
        distance_delta: float = 1e-3,
        failure_epsilon_max: float = 0.05,
        energy_eps: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Core neural mapping:
            z = (softplus(a) + delta)^(-1/2)
            phi = p + z
            rho = -log(1/2 + 2 sqrt(epsilon)) / log(2) = 1/R(epsilon)
            lambda_eff = sqrt(E_gap (2 Lambda - E_gap))
            g = rho / sqrt(1 + lambda_eff^2)
            psi = z g exp(i phi)
        """
        sp_a = QuantumResourcePhasingAnalyticalKernels.softplus(distance_raw)
        radius = sp_a + distance_delta
        inv_sqrt = 1.0 / np.sqrt(radius)
        phase = phase0 + inv_sqrt
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        reliability, failure, sqrt_failure = QuantumResourcePhasingAnalyticalKernels.dqi_reliability(
            syndrome, failure_epsilon_max
        )
        reliability_argument = 0.5 + 2.0 * sqrt_failure

        gap = QuantumResourcePhasingAnalyticalKernels.softplus(gap_raw) + energy_eps
        positive_norm = QuantumResourcePhasingAnalyticalKernels.softplus(norm_raw) + energy_eps
        spectral_norm = gap + positive_norm
        radicand = gap * (2.0 * spectral_norm - gap)
        lambda_eff = np.sqrt(radicand)
        block_gate = 1.0 / np.sqrt(1.0 + lambda_eff * lambda_eff)

        envelope = inv_sqrt * reliability * block_gate
        out_real = envelope * cos_phase
        out_imag = envelope * sin_phase

        cache = {
            "distance_raw": distance_raw,
            "phase0": phase0,
            "syndrome": syndrome,
            "gap_raw": gap_raw,
            "norm_raw": norm_raw,
            "radius": radius,
            "inv_sqrt": inv_sqrt,
            "phase": phase,
            "cos_phase": cos_phase,
            "sin_phase": sin_phase,
            "failure": failure,
            "sqrt_failure": sqrt_failure,
            "reliability_argument": reliability_argument,
            "reliability": reliability,
            "gap": gap,
            "spectral_norm": spectral_norm,
            "lambda_eff": lambda_eff,
            "block_gate": block_gate,
            "envelope": envelope,
        }
        return out_real, out_imag, cache

    @staticmethod
    def fused_resource_phasing_backward(
        grad_real: np.ndarray,
        grad_imag: np.ndarray,
        cache: Dict[str, np.ndarray],
        failure_epsilon_max: float = 0.05,
        grad_reliability: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Exact, singularity-free reverse accumulation for all five dynamic input tensors."""
        a = cache["distance_raw"]
        s = cache["syndrome"]
        u = cache["gap_raw"]
        v = cache["norm_raw"]
        radius = cache["radius"]
        z = cache["inv_sqrt"]
        cos_phase = cache["cos_phase"]
        sin_phase = cache["sin_phase"]
        sqrt_failure = cache["sqrt_failure"]
        reliability_argument = cache["reliability_argument"]
        reliability = cache["reliability"]
        gap = cache["gap"]
        spectral_norm = cache["spectral_norm"]
        lambda_eff = cache["lambda_eff"]
        block_gate = cache["block_gate"]
        envelope = cache["envelope"]

        d_envelope = grad_real * cos_phase + grad_imag * sin_phase
        d_phase = envelope * (-grad_real * sin_phase + grad_imag * cos_phase)
        d_inv_sqrt = d_envelope * reliability * block_gate + d_phase
        d_reliability_gate = d_envelope * z

        sigmoid = QuantumResourcePhasingAnalyticalKernels.sigmoid
        d_distance_raw = d_inv_sqrt * (-0.5 * z / radius) * sigmoid(a)
        d_phase0 = d_phase

        d_reliability = d_reliability_gate * block_gate
        if grad_reliability is not None:
            d_reliability = d_reliability + grad_reliability

        # Algebraically cancelled derivative (no 1/sqrt_failure division singularity):
        # d(reliability)/ds = -sqrt(failure) * (1 - sig_s) / (LN2 * (0.5 + 2 * sqrt(failure)))
        sig_s = sigmoid(s)
        d_syndrome = (
            -d_reliability
            * sqrt_failure
            * (1.0 - sig_s)
            / (LN2 * reliability_argument)
        )

        d_block_gate = d_reliability_gate * reliability
        block_gate_cubed = block_gate * block_gate * block_gate
        d_lambda = d_block_gate * (-lambda_eff * block_gate_cubed)
        d_radicand = d_lambda / (2.0 * lambda_eff)
        d_gap_raw = d_radicand * (2.0 * spectral_norm) * sigmoid(u)
        d_norm_raw = d_radicand * (2.0 * gap) * sigmoid(v)

        return {
            "d_distance_raw": d_distance_raw,
            "d_phase0": d_phase0,
            "d_syndrome": d_syndrome,
            "d_gap_raw": d_gap_raw,
            "d_norm_raw": d_norm_raw,
        }

    @staticmethod
    def pairwise_inverse_sqrt_attention_forward(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        token_reliability: np.ndarray,
        pair_lambda: np.ndarray,
        pair_gamma: np.ndarray,
        pair_distance_delta: float = 1.0,
        reliability_eps: float = 1e-8,
        causal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attention with softened pairwise inverse-distance bias and DQI source gate.

        Uses softplus(pair_lambda) and softplus(pair_gamma) to match PyTorch
        QuantumResourcePhasingOp.forward identically.

        q, k, v: [B, H, L, D_head]
        token_reliability: [B, L]
        pair_lambda, pair_gamma: [H]
        """
        _, _, seq_len, d_head = q.shape
        indices = np.arange(seq_len, dtype=np.float64)
        dt = indices[:, None] - indices[None, :]
        effective_distance = np.maximum(dt, 0.0) if causal else np.abs(dt)
        distance_squared = effective_distance ** 2

        lam = QuantumResourcePhasingAnalyticalKernels.softplus(pair_lambda)[:, None, None]
        gamma = QuantumResourcePhasingAnalyticalKernels.softplus(pair_gamma)[:, None, None]
        pair_bias = gamma / np.sqrt(pair_distance_delta + lam * distance_squared[None, :, :])

        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(d_head)
        scores = scores + pair_bias[None, :, :, :]
        source_gate = np.log(np.maximum(token_reliability, 0.0) + reliability_eps)
        scores = scores + source_gate[:, None, None, :]
        if causal:
            scores = np.where((dt >= 0)[None, None, :, :], scores, -np.inf)

        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out = np.matmul(attention, v)
        return out, attention

    @staticmethod
    def pairwise_inverse_sqrt_attention_backward(
        grad_out: np.ndarray,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        token_reliability: np.ndarray,
        pair_lambda: np.ndarray,
        pair_gamma: np.ndarray,
        attention: np.ndarray,
        pair_distance_delta: float = 1.0,
        reliability_eps: float = 1e-8,
        causal: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Exact analytical backward for pairwise_inverse_sqrt_attention_forward.

        Returns gradients with respect to q, k, v, token_reliability,
        pair_lambda, and pair_gamma.
        """
        _, _, seq_len, d_head = q.shape
        scale = 1.0 / math.sqrt(d_head)

        d_v = np.matmul(np.swapaxes(attention, -1, -2), grad_out)
        d_attention = np.matmul(grad_out, np.swapaxes(v, -1, -2))

        d_scores = attention * (
            d_attention - np.sum(attention * d_attention, axis=-1, keepdims=True)
        )
        indices = np.arange(seq_len, dtype=np.float64)
        dt = indices[:, None] - indices[None, :]
        if causal:
            d_scores = np.where((dt >= 0)[None, None, :, :], d_scores, 0.0)

        d_q = np.matmul(d_scores, k) * scale
        d_k = np.matmul(np.swapaxes(d_scores, -1, -2), q) * scale

        effective_distance = np.maximum(dt, 0.0) if causal else np.abs(dt)
        distance_squared = (effective_distance ** 2)[None, :, :]
        lam = QuantumResourcePhasingAnalyticalKernels.softplus(pair_lambda)[:, None, None]
        gamma = QuantumResourcePhasingAnalyticalKernels.softplus(pair_gamma)[:, None, None]
        radicand = pair_distance_delta + lam * distance_squared
        inv_sqrt_rad = 1.0 / np.sqrt(radicand)

        d_pair_bias = np.sum(d_scores, axis=0)  # [H, L, L]
        sig_gamma = QuantumResourcePhasingAnalyticalKernels.sigmoid(pair_gamma)
        sig_lambda = QuantumResourcePhasingAnalyticalKernels.sigmoid(pair_lambda)

        d_pair_gamma = sig_gamma * np.sum(d_pair_bias * inv_sqrt_rad, axis=(-2, -1))
        d_pair_lambda = sig_lambda * np.sum(
            d_pair_bias * (-0.5 * gamma * (inv_sqrt_rad ** 3) * distance_squared),
            axis=(-2, -1),
        )

        d_source_gate = np.sum(d_scores, axis=(1, 2))  # [B, L]
        active_rel_mask = (token_reliability >= 0.0).astype(np.float64)
        d_token_reliability = (
            d_source_gate
            * active_rel_mask
            / (np.maximum(token_reliability, 0.0) + reliability_eps)
        )

        return {
            "d_q": d_q,
            "d_k": d_k,
            "d_v": d_v,
            "d_token_reliability": d_token_reliability,
            "d_pair_lambda": d_pair_lambda,
            "d_pair_gamma": d_pair_gamma,
        }


# ==============================================================================
# 4. PyTorch autograd and neural operator
# ==============================================================================

if HAS_TORCH:

    class RMSNorm(nn.Module):
        """Root-mean-square normalization."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_float = x.to(torch.float32)
            scale = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (x_float * scale).to(x.dtype) * self.weight


    class FusedQuantumResourcePhasingFunction(torch.autograd.Function):
        """
        Custom exact autograd with fused Triton CUDA dispatch.

        Returns (out_real, out_imag, reliability) in a single fused pass so
        QuantumResourcePhasingOp.forward never recomputes reliability in eager
        mode and saves only 5 input tensors in ctx.save_for_backward.
        """

        @staticmethod
        def forward(
            ctx,
            distance_raw: torch.Tensor,
            phase0: torch.Tensor,
            syndrome: torch.Tensor,
            gap_raw: torch.Tensor,
            norm_raw: torch.Tensor,
            distance_delta: float,
            failure_epsilon_max: float,
            energy_eps: float,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            use_triton = bool(distance_raw.is_cuda and HAS_TRITON)
            ctx.use_triton = use_triton
            ctx.distance_delta = distance_delta
            ctx.failure_epsilon_max = failure_epsilon_max
            ctx.energy_eps = energy_eps
            ctx.save_for_backward(distance_raw, phase0, syndrome, gap_raw, norm_raw)

            if use_triton:
                distance_raw = distance_raw.contiguous()
                phase0 = phase0.contiguous()
                syndrome = syndrome.contiguous()
                gap_raw = gap_raw.contiguous()
                norm_raw = norm_raw.contiguous()
                out_real = torch.empty_like(distance_raw, memory_format=torch.contiguous_format)
                out_imag = torch.empty_like(distance_raw, memory_format=torch.contiguous_format)
                out_reliability = torch.empty_like(distance_raw, memory_format=torch.contiguous_format)
                d_model = distance_raw.shape[-1]
                num_rows = distance_raw.numel() // d_model
                block_d = min(triton.next_power_of_2(d_model), 1024)
                grid = (triton.cdiv(num_rows, 1),)
                fused_quantum_resource_phasing_fwd_kernel[grid](
                    distance_raw, phase0, syndrome, gap_raw, norm_raw,
                    out_real, out_imag, out_reliability,
                    d_model,
                    D_MODEL=d_model,
                    DISTANCE_DELTA=distance_delta,
                    FAILURE_MAX=failure_epsilon_max,
                    ENERGY_EPS=energy_eps,
                    LN_2=LN2,
                    BLOCK_D=block_d,
                    IS_FP64=(distance_raw.dtype == torch.float64),
                )
                return out_real, out_imag, out_reliability

            sp_a = F.softplus(distance_raw)
            radius = sp_a + distance_delta
            inv_sqrt = torch.rsqrt(radius)
            phase = phase0 + inv_sqrt
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)

            sig_s = torch.sigmoid(syndrome)
            failure = failure_epsilon_max * sig_s
            sqrt_failure = torch.sqrt(failure)
            reliability_argument = 0.5 + 2.0 * sqrt_failure
            reliability = -torch.log(reliability_argument) / LN2

            gap = F.softplus(gap_raw) + energy_eps
            positive_norm = F.softplus(norm_raw) + energy_eps
            spectral_norm = gap + positive_norm
            lambda_eff = torch.sqrt(gap * (2.0 * spectral_norm - gap))
            block_gate = torch.rsqrt(1.0 + lambda_eff * lambda_eff)

            envelope = inv_sqrt * reliability * block_gate
            out_real = envelope * cos_phase
            out_imag = envelope * sin_phase

            return out_real, out_imag, reliability

        @staticmethod
        def backward(
            ctx,
            grad_real: torch.Tensor,
            grad_imag: torch.Tensor,
            grad_reliability: Optional[torch.Tensor] = None,
        ):
            distance_raw, phase0, syndrome, gap_raw, norm_raw = ctx.saved_tensors
            if grad_reliability is None:
                grad_reliability = torch.zeros_like(grad_real)

            if ctx.use_triton:
                grad_real = grad_real.contiguous()
                grad_imag = grad_imag.contiguous()
                grad_reliability = grad_reliability.contiguous()
                distance_raw = distance_raw.contiguous()
                phase0 = phase0.contiguous()
                syndrome = syndrome.contiguous()
                gap_raw = gap_raw.contiguous()
                norm_raw = norm_raw.contiguous()
                grad_distance = torch.empty_like(distance_raw, memory_format=torch.contiguous_format)
                grad_phase = torch.empty_like(phase0, memory_format=torch.contiguous_format)
                grad_syndrome = torch.empty_like(syndrome, memory_format=torch.contiguous_format)
                grad_gap = torch.empty_like(gap_raw, memory_format=torch.contiguous_format)
                grad_norm = torch.empty_like(norm_raw, memory_format=torch.contiguous_format)
                d_model = distance_raw.shape[-1]
                num_rows = distance_raw.numel() // d_model
                block_d = min(triton.next_power_of_2(d_model), 1024)
                grid = (triton.cdiv(num_rows, 1),)
                fused_quantum_resource_phasing_bwd_kernel[grid](
                    grad_real, grad_imag, grad_reliability,
                    distance_raw, phase0, syndrome, gap_raw, norm_raw,
                    grad_distance, grad_phase, grad_syndrome, grad_gap, grad_norm,
                    d_model,
                    D_MODEL=d_model,
                    DISTANCE_DELTA=ctx.distance_delta,
                    FAILURE_MAX=ctx.failure_epsilon_max,
                    ENERGY_EPS=ctx.energy_eps,
                    LN_2=LN2,
                    BLOCK_D=block_d,
                    IS_FP64=(distance_raw.dtype == torch.float64),
                )
                return (
                    grad_distance, grad_phase, grad_syndrome, grad_gap, grad_norm,
                    None, None, None,
                )

            a, p, s, u, v = distance_raw, phase0, syndrome, gap_raw, norm_raw
            radius = F.softplus(a) + ctx.distance_delta
            z = torch.rsqrt(radius)
            phase = p + z
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)

            sig_s = torch.sigmoid(s)
            failure = ctx.failure_epsilon_max * sig_s
            sqrt_failure = torch.sqrt(failure)
            reliability_argument = 0.5 + 2.0 * sqrt_failure
            reliability = -torch.log(reliability_argument) / LN2

            gap = F.softplus(u) + ctx.energy_eps
            spectral_norm = gap + F.softplus(v) + ctx.energy_eps
            lambda_eff = torch.sqrt(gap * (2.0 * spectral_norm - gap))
            block_gate = torch.rsqrt(1.0 + lambda_eff * lambda_eff)
            envelope = z * reliability * block_gate

            d_envelope = grad_real * cos_phase + grad_imag * sin_phase
            d_phase = envelope * (-grad_real * sin_phase + grad_imag * cos_phase)
            d_inv_sqrt = d_envelope * reliability * block_gate + d_phase
            d_reliability_gate = d_envelope * z

            d_a = d_inv_sqrt * (-0.5 * z / radius) * torch.sigmoid(a)
            d_p = d_phase

            # Singularity-free cancelled derivative:
            d_reliability = d_reliability_gate * block_gate + grad_reliability
            d_s = (
                -d_reliability
                * sqrt_failure
                * (1.0 - sig_s)
                / (LN2 * reliability_argument)
            )

            d_block_gate = d_reliability_gate * reliability
            block_gate_cubed = block_gate * block_gate * block_gate
            d_lambda = d_block_gate * (-lambda_eff * block_gate_cubed)
            d_radicand = d_lambda / (2.0 * lambda_eff)
            d_u = d_radicand * (2.0 * spectral_norm) * torch.sigmoid(u)
            d_v = d_radicand * (2.0 * gap) * torch.sigmoid(v)

            return d_a, d_p, d_s, d_u, d_v, None, None, None


    class QuantumResourcePhasingOp(nn.Module):
        """
        Complete quantum-resource-inspired neural feature operator.

        Local path:
            inverse-square-root phasing -> DQI reliability -> block-encoding gate.
        Sequence path:
            causal/full attention with softened pairwise inverse-distance bias.
        """

        def __init__(self, config: QuantumResourcePhasingConfig):
            super().__init__()
            self.config = config
            self.d_model = config.d_model
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads

            self.state_proj = nn.Linear(config.d_model, 5 * config.d_model, bias=True)
            self.qkv_pair = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            self.pair_lambda = nn.Parameter(torch.full((config.n_heads,), 0.1))
            self.pair_gamma = nn.Parameter(torch.full((config.n_heads,), 0.2))
            self.collapse_proj = nn.Linear(3 * config.d_model, config.d_model, bias=False)
            self.out_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            if x.ndim != 3 or x.shape[-1] != self.d_model:
                raise ValueError("x must have shape [B, L, d_model].")
            batch, seq_len, d_model = x.shape
            raw = self.state_proj(x)
            distance_raw, phase0, syndrome, gap_raw, norm_raw = raw.chunk(5, dim=-1)

            psi_real, psi_imag, reliability = FusedQuantumResourcePhasingFunction.apply(
                distance_raw, phase0, syndrome, gap_raw, norm_raw,
                self.config.distance_delta,
                self.config.failure_epsilon_max,
                self.config.energy_eps,
            )

            qkv = self.qkv_pair(x).view(
                batch, seq_len, 3, self.n_heads, self.d_head
            )
            q, k, v = [part.transpose(1, 2) for part in qkv.unbind(dim=2)]

            indices = torch.arange(seq_len, device=x.device)
            dt = indices[:, None] - indices[None, :]
            effective_dt = dt.clamp(min=0) if self.config.mixing_mode == "causal" else dt.abs()
            effective_dt = effective_dt.to(x.dtype)
            pair_lambda = F.softplus(self.pair_lambda).view(1, self.n_heads, 1, 1)
            pair_gamma = F.softplus(self.pair_gamma).view(1, self.n_heads, 1, 1)
            pair_bias = pair_gamma / torch.sqrt(
                self.config.pair_distance_delta
                + pair_lambda * effective_dt.view(1, 1, seq_len, seq_len) ** 2
            )

            token_reliability = reliability.mean(dim=-1)

            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_head ** -0.5)
            scores = scores + pair_bias
            scores = scores + torch.log(
                token_reliability.clamp_min(0.0) + self.config.reliability_eps
            )[:, None, None, :]
            if self.config.mixing_mode == "causal":
                causal_mask = dt >= 0
                scores = scores.masked_fill(
                    ~causal_mask.view(1, 1, seq_len, seq_len), float("-inf")
                )
            attention = F.softmax(scores, dim=-1)
            pair_context = torch.matmul(attention, v)
            pair_context = pair_context.transpose(1, 2).contiguous().view(
                batch, seq_len, d_model
            )

            combined = torch.cat([psi_real, psi_imag, pair_context], dim=-1)
            out = self.out_norm(self.collapse_proj(combined))

            with torch.no_grad():
                inv_sqrt = torch.rsqrt(F.softplus(distance_raw) + self.config.distance_delta)
                failure = self.config.failure_epsilon_max * torch.sigmoid(syndrome)
                gap = F.softplus(gap_raw) + self.config.energy_eps
                spectral_norm = gap + F.softplus(norm_raw) + self.config.energy_eps
                lambda_eff = torch.sqrt(gap * (2.0 * spectral_norm - gap))
                diagnostics = {
                    "mean_inverse_sqrt": float(inv_sqrt.mean().item()),
                    "mean_failure_proxy": float(failure.mean().item()),
                    "mean_dqi_reliability": float(reliability.mean().item()),
                    "mean_lambda_eff": float(lambda_eff.mean().item()),
                    "mean_pair_attention_entropy": float(
                        (-(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log())
                         .sum(dim=-1).mean()).item()
                    ),
                }
            return out, diagnostics


# ==============================================================================
# 5. Tile simulator and standalone verification
# ==============================================================================

def simulate_triton_quantum_resource_kernel(
    distance_raw: np.ndarray,
    phase0: np.ndarray,
    syndrome: np.ndarray,
    gap_raw: np.ndarray,
    norm_raw: np.ndarray,
    distance_delta: float = 1e-3,
    failure_epsilon_max: float = 0.05,
    energy_eps: float = 1e-3,
    block_d: int = 64,
    bytes_per_elem: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Tile-by-tile NumPy simulator of the column-tiled fused Triton forward kernel."""
    batch, seq_len, d_model = distance_raw.shape
    flat_inputs = [x.reshape(batch * seq_len, d_model) for x in (
        distance_raw, phase0, syndrome, gap_raw, norm_raw
    )]
    real_flat = np.zeros((batch * seq_len, d_model), dtype=np.float64)
    imag_flat = np.zeros_like(real_flat)

    for d0 in range(0, d_model, block_d):
        d1 = min(d0 + block_d, d_model)
        tile_inputs = [x[:, d0:d1] for x in flat_inputs]
        tile_real, tile_imag, _ = QuantumResourcePhasingAnalyticalKernels.fused_resource_phasing_forward(
            *tile_inputs,
            distance_delta=distance_delta,
            failure_epsilon_max=failure_epsilon_max,
            energy_eps=energy_eps,
        )
        real_flat[:, d0:d1] = tile_real
        imag_flat[:, d0:d1] = tile_imag

    out_shape = (batch, seq_len, d_model)
    out_real = real_flat.reshape(out_shape)
    out_imag = imag_flat.reshape(out_shape)

    element_bytes = batch * seq_len * d_model * bytes_per_elem
    eager_hbm_bytes = 30 * element_bytes
    # 5 input tensors read + 3 output tensors (out_real, out_imag, out_reliability) written
    fused_hbm_bytes = 8 * element_bytes
    metrics = {
        "eager_hbm_kb": eager_hbm_bytes / 1024.0,
        "fused_hbm_kb": fused_hbm_bytes / 1024.0,
        "hbm_saved_pct": (1.0 - fused_hbm_bytes / eager_hbm_bytes) * 100.0,
    }
    return out_real, out_imag, metrics


def run_quantum_resource_phasing_verification_suite() -> None:
    """Run finite-difference, extreme-underflow stress, tile-parity, and pairwise attention checks."""
    rng = np.random.default_rng(20261105)
    print("=" * 84)
    print("1. NumPy analytical backward vs. central finite differences (including grad_reliability)")
    print("=" * 84)

    shape = (2, 3, 4)
    inputs = {
        "distance_raw": rng.standard_normal(shape) * 0.4,
        "phase0": rng.standard_normal(shape) * 0.5,
        "syndrome": rng.standard_normal(shape) * 0.4,
        "gap_raw": rng.standard_normal(shape) * 0.3,
        "norm_raw": rng.standard_normal(shape) * 0.3,
    }
    grad_real = rng.standard_normal(shape)
    grad_imag = rng.standard_normal(shape)
    grad_rel = rng.standard_normal(shape) * 0.2
    kwargs = {
        "distance_delta": 1e-3,
        "failure_epsilon_max": 0.05,
        "energy_eps": 1e-3,
    }

    def loss_from(inputs_dict: Dict[str, np.ndarray]) -> float:
        real, imag, c = QuantumResourcePhasingAnalyticalKernels.fused_resource_phasing_forward(
            **inputs_dict, **kwargs
        )
        return float(np.sum(real * grad_real + imag * grad_imag + c["reliability"] * grad_rel))

    _, _, cache = QuantumResourcePhasingAnalyticalKernels.fused_resource_phasing_forward(
        **inputs, **kwargs
    )
    analytical = QuantumResourcePhasingAnalyticalKernels.fused_resource_phasing_backward(
        grad_real, grad_imag, cache, failure_epsilon_max=0.05, grad_reliability=grad_rel
    )

    finite_difference_eps = 1e-6
    for name, array in inputs.items():
        numerical = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + finite_difference_eps
            loss_plus = loss_from(inputs)
            array[index] = original - finite_difference_eps
            loss_minus = loss_from(inputs)
            array[index] = original
            numerical[index] = (loss_plus - loss_minus) / (2.0 * finite_difference_eps)
        gradient = analytical["d_" + name]
        relative_error = np.linalg.norm(gradient - numerical) / (
            np.linalg.norm(gradient) + np.linalg.norm(numerical) + 1e-15
        )
        print(f"   dL/d({name:12s}) relative L2 error: {relative_error:.3e}")
        assert relative_error < 1e-7, f"Gradient check failed for {name}."

    print("\n" + "=" * 84)
    print("2. Extreme negative syndrome underflow stress test (s = -90 in fp32, s = -750 in fp64)")
    print("=" * 84)
    stress_s = np.array([[[-90.0, -750.0, 0.0, 50.0]]], dtype=np.float64)
    stress_zeros = np.zeros_like(stress_s)
    _, _, stress_cache = QuantumResourcePhasingAnalyticalKernels.fused_resource_phasing_forward(
        stress_zeros, stress_zeros, stress_s, stress_zeros, stress_zeros, **kwargs
    )
    stress_grads = QuantumResourcePhasingAnalyticalKernels.fused_resource_phasing_backward(
        np.ones_like(stress_s), np.ones_like(stress_s), stress_cache,
        failure_epsilon_max=0.05, grad_reliability=np.ones_like(stress_s)
    )
    assert all(np.all(np.isfinite(g)) for g in stress_grads.values()), "Non-finite gradient under extreme s!"
    assert stress_grads["d_syndrome"][0, 0, 1] == 0.0
    print(f"   d_syndrome at s=[-90, -750, 0, 50]: {stress_grads['d_syndrome'].ravel()}")
    print("   Zero-NaN / Zero-Inf underflow safety: PASS")

    print("\n" + "=" * 84)
    print("3. Tile simulator parity")
    print("=" * 84)
    reference_real, reference_imag, _ = QuantumResourcePhasingAnalyticalKernels.fused_resource_phasing_forward(
        **inputs, **kwargs
    )
    simulated_real, simulated_imag, traffic = simulate_triton_quantum_resource_kernel(
        **inputs, **kwargs, block_d=3
    )
    tile_error = max(
        float(np.max(np.abs(reference_real - simulated_real))),
        float(np.max(np.abs(reference_imag - simulated_imag))),
    )
    print(f"   Maximum tile difference: {tile_error:.3e}")
    print(f"   Estimated HBM traffic reduction: {traffic['hbm_saved_pct']:.1f}%")
    assert tile_error == 0.0

    print("\n" + "=" * 84)
    print("4. Pairwise attention forward/backward gradcheck and PyTorch module")
    print("=" * 84)
    attn_inputs = {
        "q": rng.standard_normal((2, 2, 3, 4)) * 0.3,
        "k": rng.standard_normal((2, 2, 3, 4)) * 0.3,
        "v": rng.standard_normal((2, 2, 3, 4)) * 0.3,
        "token_reliability": rng.uniform(0.1, 0.9, size=(2, 3)),
        "pair_lambda": np.array([0.1, -0.3], dtype=np.float64),
        "pair_gamma": np.array([0.2, -0.1], dtype=np.float64),
    }
    grad_attn_out = rng.standard_normal((2, 2, 3, 4))
    attention_out, attention = QuantumResourcePhasingAnalyticalKernels.pairwise_inverse_sqrt_attention_forward(
        **attn_inputs, causal=True
    )
    assert attention_out.shape == attn_inputs["q"].shape
    np.testing.assert_allclose(attention.sum(axis=-1), 1.0, atol=1e-12)
    assert np.allclose(attention[:, :, 0, 1:], 0.0)

    attn_grads = QuantumResourcePhasingAnalyticalKernels.pairwise_inverse_sqrt_attention_backward(
        grad_attn_out, **attn_inputs, attention=attention, causal=True
    )

    def attn_loss_from(d: Dict[str, np.ndarray]) -> float:
        out_val, _ = QuantumResourcePhasingAnalyticalKernels.pairwise_inverse_sqrt_attention_forward(
            **d, causal=True
        )
        return float(np.sum(out_val * grad_attn_out))

    for name, array in attn_inputs.items():
        numerical = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + finite_difference_eps
            loss_plus = attn_loss_from(attn_inputs)
            array[index] = original - finite_difference_eps
            loss_minus = attn_loss_from(attn_inputs)
            array[index] = original
            numerical[index] = (loss_plus - loss_minus) / (2.0 * finite_difference_eps)
        gradient = attn_grads["d_" + name]
        relative_error = np.linalg.norm(gradient - numerical) / (
            np.linalg.norm(gradient) + np.linalg.norm(numerical) + 1e-15
        )
        print(f"   Attention dL/d({name:17s}) relative L2 error: {relative_error:.3e}")
        assert relative_error < 1e-7, f"Attention gradient check failed for {name}."
    print("   NumPy causal pairwise attention forward + backward: PASS")

    if HAS_TORCH:
        torch_inputs = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs.values()]
        torch_grad_real = torch.tensor(grad_real, dtype=torch.float64)
        torch_grad_imag = torch.tensor(grad_imag, dtype=torch.float64)
        torch_grad_rel = torch.tensor(grad_rel, dtype=torch.float64)

        def torch_function(*args):
            real, imag, rel = FusedQuantumResourcePhasingFunction.apply(
                *args, 1e-3, 0.05, 1e-3
            )
            return (
                (real * torch_grad_real).sum()
                + (imag * torch_grad_imag).sum()
                + (rel * torch_grad_rel).sum()
            )

        torch_gradcheck = torch.autograd.gradcheck(
            torch_function,
            tuple(torch_inputs),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        assert torch_gradcheck

        config = QuantumResourcePhasingConfig(d_model=16, n_heads=4)
        operator = QuantumResourcePhasingOp(config).double()
        x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        out, diagnostics = operator(x)
        assert out.shape == x.shape
        loss = out.square().mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in operator.parameters() if p.grad is not None)
        print("   PyTorch custom-autograd gradcheck: PASS")
        print("   Full operator forward/backward: PASS")
        print(f"   Output shape: {tuple(out.shape)}")
        print(f"   Diagnostics: {diagnostics}")
    else:
        print("   PyTorch unavailable: skipped torch checks; NumPy path remains usable.")

    print("\nAll quantum resource phasing operator checks passed.")


if not HAS_TORCH:
    class RMSNorm:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate RMSNorm.")

    class FusedQuantumResourcePhasingFunction:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for the custom autograd function.")

    class QuantumResourcePhasingOp:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required to instantiate QuantumResourcePhasingOp; "
                "use QuantumResourcePhasingAnalyticalKernels for NumPy execution."
            )



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
    run_quantum_resource_phasing_verification_suite()
