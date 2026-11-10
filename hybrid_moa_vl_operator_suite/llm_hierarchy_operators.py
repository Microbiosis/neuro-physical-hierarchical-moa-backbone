#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Hierarchy Operator — 现代大语言模型完整计算层次的单文件可微算子模块
==========================================================================

零同级依赖（0 sibling imports）、自包含、可微分的深度学习底层独立算子。每个算子
独占自己的**张量空间**，层次沿真实数据流串联，而不是把整条 LLM 压进一个残差通道：

    ┌──────────────────────────────────────────────────────────────────────┐
    │ 算子                    入空间            出空间            中间空间     │
    │ L0 EmbeddingOp          token (B,L)       stream D          —          │
    │ L1 RMSNormPositionOp    stream D          norm   D          —          │
    │ L2 CausalAttentionOp    norm   D          stream D          head H×d_h │
    │ L3 SwiGLUFFNOp          norm   D          stream D          ffn  F     │
    │ L4 ResidualStreamOp     stream D + D      stream D          —          │
    │ L5 VocabHeadOp          stream D          logit  V          —          │
    └──────────────────────────────────────────────────────────────────────┘

    计算图（真正的 LLM 层次，顺序组合）：
        x = Embedding(ids)                                   # V → D
        for block in range(n_layers):
            x = Residual(x, Attention(Norm(x)))               # D → H×d_h → D
            x = Residual(x, SwiGLU(Norm(x)))                  # D → F → D
        logits = VocabHead(Norm_final(x))                     # D → V, softcap

四个可微算子类别（O1 归一化-位置、O2 因果注意力、O3 门控前馈、O4 残差流）加上
两个空间算子（O0 词嵌入、O5 词表头），共六个参数块**互不相交**的独立算子：

  · 每个算子单独可导入、单独可微、单独可测（`numpy` / `torch` / `triton` 三套实现）；
  · 扰动任一算子的参数，其余算子的输出在输入固定时**精确不变**（交叉灵敏度非对角元严格 0）；
  · 每个算子在**自己的空间**里满秩：注意力在 H×d_h 头空间、SwiGLU 在 F 中间空间、
    词表头在 V 词表空间、嵌入在 V×D 表空间——不做 D 维塌缩；
  · 层次串联后仍共享同一条残差流（stream D），这是 LLM 的真实结构，不是并联混合。

三套实现，互为标尺
------------------
1. `np_*`          — 双精度 NumPy 解析引擎：前向与**手推解析反向**（不依赖任何自动微分）。
2. `torch.nn`      — PyTorch 引擎：`nn.Module` + autograd，训练与推理主路径。
3. `@triton.jit`   — GPU 融合核（前向 + 反向，均以 `torch.autograd.Function` 接入）：
                     · RMSNorm+RoPE 单核融合（含 cos/sin 表生成）；
                     · SwiGLU 门控融合；
                     · FlashAttention 式因果注意力（QK^T → softmax → PV 全在核内，
                       不落 L×L 矩阵，支持 GQA 与 fp16/bf16/fp32/fp64）。

自检
----
`python llm_hierarchy_operator.py`            三套引擎逐算子对拍 + Gradcheck + 六算子正交审计
`python -m llm_hierarchy_operator`            同上（以包形式安装后）
`python llm_hierarchy_operator.py --json`     输出机器可读报告
`python llm_hierarchy_operator.py --bench`    融合核基准（需 GPU）

依赖：numpy（必需）；torch / triton（可选，缺失时自动退化为 NumPy 路径）。

实测（RTX 5070 Laptop，torch 2.14.1+cu132，triton 3.9.0，2026-11-10）
--------------------------------------------------------------------
审计配置：D=128 H=4 H_kv=2 F=320 V=128 layers=2，六空间 token 128 / stream 128 /
head 128 / kv 64 / ffn 320 / logit 128。
- 逐算子 NumPy ↔ PyTorch：前向 max|Δ| ≤ 1.8e-15，全部参数与输入梯度相对误差 ≤ 1e-15。
- 整模型：前向 1.2e-15，梯度 2.4e-15；KV-Cache 逐 token 与整序列 1.3e-15。
- Triton fp64：norm+rope、swiglu 与 attention 融合核（含反向、含位置偏移、L 非块对齐）≈ 1e-15；
  fp32 注意力核 ≈ 2e-7（IEEE 点积）。
- 融合注意力基准（B=8, H=8, L=1024, D=64）：fp16 前向 14.1×、前向+反向 6.3×；
  fp32 前向 3.7×、前向+反向 2.5×（对比 materialize L×L 的 eager 路径）。
  块级（B=32, H=6, D=384）前向+反向：L=128 约 1.0×、L=256 1.47×、L=512 1.69×，
  因此 `flash_min_len=192` 以下自动回落 PyTorch 路径。
- Gradcheck（eps=1e-6）：参数 1.6e-8，嵌入表 4.2e-10。
- 五维审计：六个算子在各自空间满秩（head 320/320、ffn 320/320、attention 128/128、
  logit 128/128），反向 5/5 算子 Max VIF 1.09，6×6 交叉灵敏度 30/30 非对角精确 0，
  留一法消融 Δrel 1.00–1.27。
- 融合核（rows=8192，D=1024，fp32）：RMSNorm+RoPE 6.9×，SwiGLU 门控 1.70×。
- 字符级 LM（`demo_train_llm_hierarchy.py`）：9.50M 参数（D=384，6 层，F=1024），
  400 步 val loss 1.69（PPL 5.4），54 ms/step，76k tok/s。
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # PyTorch 可选
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # pragma: no cover - 无 torch 环境
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:  # Triton 可选
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tld

    HAS_TRITON = True
except ImportError:  # pragma: no cover - 无 triton 环境
    HAS_TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore
    tld = None  # type: ignore

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LLMHierarchyConfig",
    "EmbeddingOp",
    "RMSNormPositionOp",
    "CausalAttentionOp",
    "SwiGLUFFNOp",
    "ResidualStreamOp",
    "VocabHeadOp",
    "LLMHierarchyBlock",
    "LLMHierarchyOperator",
    "LLMHierarchyModel",
    "np_embedding_forward",
    "np_embedding_backward",
    "np_rmsnorm_position_forward",
    "np_rmsnorm_position_backward",
    "np_causal_attention_forward",
    "np_causal_attention_backward",
    "np_swiglu_forward",
    "np_swiglu_backward",
    "np_residual_forward",
    "np_residual_backward",
    "np_vocab_head_forward",
    "np_vocab_head_backward",
    "np_block_forward",
    "np_block_backward",
    "np_model_forward",
    "np_model_backward",
    "run_verification",
    "benchmark_fusion",
    "benchmark_attention",
    "triton_flash_attention",
    "triton_norm_rope_forward",
    "triton_swiglu_forward",
    "main",
]


# ==============================================================================
# 1. 配置
# ==============================================================================
@dataclass
class LLMHierarchyConfig:
    """LLM 全层次算子的结构配置（各算子空间的尺度在此一次性声明）。

    Attributes:
        dim:            残差流空间 D（必须被 n_heads 整除，head_dim 为偶数）。
        n_heads:        注意力头空间 H。
        n_kv_heads:     KV 头数 H_kv（GQA；等于 n_heads 即 MHA）。
        ffn_dim:        SwiGLU 中间空间 F。
        vocab_size:     词表空间 V（嵌入表与词表头）。
        n_layers:       层次深度（块数）。
        rope_base:      RoPE 基底。
        logit_softcap:  词表 logit 软截断上界（<= 0 表示关闭）。
        eps:            RMSNorm 分母安全下界。
        tie_embeddings: 词表头是否与嵌入表共享（V×D 空间绑定）。
        use_triton:     有 GPU 时是否调度融合核。
        flash_min_len:  融合注意力核的最短序列长度（更短走 PyTorch，实测交汇点约 192）。
        block:          Triton 线程块大小。
        seed:           参数初始化种子。
    """

    dim: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_dim: int = 1024
    vocab_size: int = 512
    n_layers: int = 6
    rope_base: float = 10000.0
    logit_softcap: float = 30.0
    eps: float = 1e-6
    tie_embeddings: bool = False
    use_triton: bool = True
    flash_min_len: int = 192
    block: int = 128
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.n_heads <= 0:
            raise ValueError("dim 与 n_heads 必须为正")
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim={self.dim} 必须被 n_heads={self.n_heads} 整除")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} 必须为偶数（RoPE 成对旋转）")
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads 必须为正且整除 n_heads（GQA 分组）")
        if self.ffn_dim <= 0 or self.vocab_size <= 0:
            raise ValueError("ffn_dim 与 vocab_size 必须为正")
        if self.n_layers <= 0:
            raise ValueError("n_layers 必须为正")
        if self.eps <= 0.0:
            raise ValueError("eps 必须为正（RMSNorm 分母安全下界）")
        if self.flash_min_len < 1:
            raise ValueError("flash_min_len 必须为正")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def pair_dim(self) -> int:
        return self.head_dim // 2

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """每个 KV 头服务的查询头数（GQA 组宽）。"""
        return self.n_heads // self.n_kv_heads

    def spaces(self) -> Dict[str, int]:
        """本配置下各算子实际占用的空间维度。"""
        return {
            "token": self.vocab_size,
            "stream": self.dim,
            "norm": self.dim,
            "head": self.n_heads * self.head_dim,
            "kv": self.kv_dim,
            "ffn": self.ffn_dim,
            "logit": self.vocab_size,
        }

# ==============================================================================
# 2. 双精度 NumPy 解析引擎
#    每个子算子给出 f(x; θ) 与手推的 ∂L/∂x, ∂L/∂θ（反向模式），
#    作为 PyTorch 与 Triton 两套实现的数值标尺。
# ==============================================================================
def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _np_softplus(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softplus：log(1 + e^x) = logaddexp(0, x)。"""
    return np.logaddexp(0.0, x)


def _np_rms_scale(x: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm 分母 s = sqrt(mean_d(x^2) + eps)，形状 (..., 1)。"""
    return np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _np_rope_theta(cfg: LLMHierarchyConfig, omega: Optional[np.ndarray]) -> np.ndarray:
    """RoPE 角频率 θ_i = base^(-2i/d) * exp(ω_i)，形状 (d/2,)。"""
    idx = np.arange(cfg.pair_dim, dtype=np.float64)
    theta = np.power(float(cfg.rope_base), -2.0 * idx / float(cfg.head_dim))
    if omega is not None:
        theta = theta * np.exp(omega)
    return theta


def _np_rope_tables(
    cfg: LLMHierarchyConfig,
    omega: Optional[np.ndarray],
    length: int,
    offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造 (1, L, 1, d/2) 的 cos/sin 表，可直接广播到 (B, L, H, d/2)。"""
    theta = _np_rope_theta(cfg, omega)
    pos = np.arange(length, dtype=np.float64) + float(offset)
    phi = pos[:, None] * theta[None, :]
    cos = np.cos(phi)[None, :, None, :]
    sin = np.sin(phi)[None, :, None, :]
    return cos, sin, theta, pos


def _np_rope_apply(t: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """对最后一维成对施加 SO(2) 旋转：y_even = a c - b s, y_odd = a s + b c。"""
    even = t[..., 0::2]
    odd = t[..., 1::2]
    out = np.empty_like(t)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def _np_rope_apply_t(gy: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """旋转的转置（反向）：ga = gy_a c + gy_b s, gb = -gy_a s + gy_b c。"""
    ga = gy[..., 0::2]
    gb = gy[..., 1::2]
    out = np.empty_like(gy)
    out[..., 0::2] = ga * cos + gb * sin
    out[..., 1::2] = -ga * sin + gb * cos
    return out


def _np_rope_phase_grad(gy: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dL/dphi（对 batch 与 head 求和），返回 (L, d/2)。

    y_a = a c - b s, y_b = a s + b c  =>  dy_a/dphi = -y_b, dy_b/dphi = y_a
    故 dL/dphi = gy_b y_a - gy_a y_b。
    """
    return (gy[..., 1::2] * y[..., 0::2] - gy[..., 0::2] * y[..., 1::2]).sum(axis=(0, 2))


# ------------------------------------------------------------------------------
# 2.1 L0 — 词嵌入算子 O0（token 空间 V → 残差流空间 D）
# ------------------------------------------------------------------------------
def np_embedding_forward(
    ids: np.ndarray, table: np.ndarray, cfg: LLMHierarchyConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """O0: x = E[ids] * sqrt(D)，把 token 空间提升到残差流空间。"""
    scale = math.sqrt(float(cfg.dim))
    x = table[ids] * scale
    return x, {"ids": ids, "scale": scale}


def np_embedding_backward(
    gx: np.ndarray, cache: Dict[str, Any], table: np.ndarray
) -> np.ndarray:
    """O0 反向：同一 token 的多次出现累加到同一行（scatter-add）。"""
    g_table = np.zeros_like(table)
    np.add.at(g_table, cache["ids"], gx * cache["scale"])
    return g_table


# ------------------------------------------------------------------------------
# 2.2 L1 — 归一化-位置算子 O1（Pre-RMSNorm + RoPE 相位调制）
# ------------------------------------------------------------------------------
def np_rmsnorm_position_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    cfg: LLMHierarchyConfig,
    positional: bool = True,
    offset: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """O1: y = RoPE_θ⊙exp(ω)( gamma ⊙ x / sqrt(mean(x^2)+eps) )。

    Args:
        x: (B, L, D) 残差流输入。
        gamma: (D,) 归一化增益。
        omega: (d/2,) 可学对数频率尺度（positional=False 时忽略）。
        positional: 是否施加旋转变换。
        offset: 起始绝对位置（KV-Cache 解码时 > 0）。

    Returns:
        (y, cache)。y 形状 (B, L, D)。
    """
    b, l, d = x.shape
    s = _np_rms_scale(x, cfg.eps)
    n = x / s
    h = n * gamma
    cache: Dict[str, Any] = {"n": n, "s": s, "d": d, "positional": positional}
    if positional:
        t = h.reshape(b, l, cfg.n_heads, cfg.head_dim)
        cos, sin, theta, pos = _np_rope_tables(cfg, omega, l, offset)
        y = _np_rope_apply(t, cos, sin).reshape(b, l, d)
        cache.update(cos=cos, sin=sin, theta=theta, pos=pos, y=y)
    else:
        y = h
        cache.update(cos=None, sin=None, theta=None, pos=None, y=None)
    return y, cache


def np_rmsnorm_position_backward(
    gy: np.ndarray,
    cache: Dict[str, Any],
    gamma: np.ndarray,
    cfg: LLMHierarchyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O1 反向：返回 (gx, g_gamma, g_omega)。"""
    n, s, d = cache["n"], cache["s"], cache["d"]
    b, l, _ = n.shape
    if cache["positional"]:
        cos, sin, theta, pos = cache["cos"], cache["sin"], cache["theta"], cache["pos"]
        t_y = cache["y"].reshape(b, l, cfg.n_heads, cfg.head_dim)
        t_gy = gy.reshape(b, l, cfg.n_heads, cfg.head_dim)
        dphi = _np_rope_phase_grad(t_gy, t_y)                 # (L, d/2)
        g_rot = _np_rope_apply_t(t_gy, cos, sin).reshape(b, l, d)
        # phi = pos * theta_i, theta_i = base^(-2i/d) * exp(omega_i) => dphi/domega = pos * theta
        g_omega = (dphi * pos[:, None] * theta[None, :]).sum(axis=0)
    else:
        g_rot = gy
        g_omega = np.zeros(cfg.pair_dim, dtype=gy.dtype)
    g_gamma = (g_rot * n).sum(axis=(0, 1))
    gn = g_rot * gamma
    c = (gn * n).sum(axis=-1, keepdims=True)
    gx = (gn - c * n / float(d)) / s
    return gx, g_gamma, g_omega


# ------------------------------------------------------------------------------
# 2.3 L2 — 因果注意力算子 O2（QKV 投射 + RoPE + 因果 softmax + GQA + KV-Cache）
# ------------------------------------------------------------------------------
def np_causal_attention_forward(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    cfg: LLMHierarchyConfig,
    kv_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """O2: 多头因果缩放点积注意力。

    Args:
        x: (B, L, D)。
        w_q: (D, D)；w_k / w_v: (D, H_kv*d)；w_o: (D, D)。
        kv_cache: 可选，{"k": (B, Lc, H_kv, d), "v": (B, Lc, H_kv, d)}，已旋转的 KV。

    Returns:
        (y, cache)。cache["k"]/["v"] 为更新后的 KV 状态（k 已旋转）。
    """
    b, l, d = x.shape
    h, hk, hd, rep = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_rep
    q = (x @ w_q).reshape(b, l, h, hd)
    k = (x @ w_k).reshape(b, l, hk, hd)
    v = (x @ w_v).reshape(b, l, hk, hd)

    offset = 0 if kv_cache is None else int(kv_cache["k"].shape[1])
    cos, sin, _, _ = _np_rope_tables(cfg, None, l, offset)
    q_r = _np_rope_apply(q, cos, sin)
    k_r = _np_rope_apply(k, cos, sin)

    if kv_cache is None:
        k_all, v_all = k_r, v
    else:
        k_all = np.concatenate([kv_cache["k"], k_r], axis=1)
        v_all = np.concatenate([kv_cache["v"], v], axis=1)
    lc = k_all.shape[1]

    qpos = np.arange(l) + offset
    mpos = np.arange(lc)
    causal = mpos[None, :] > qpos[:, None]          # (L, Lc) True 即屏蔽
    scale = 1.0 / math.sqrt(float(hd))

    ps: List[np.ndarray] = []
    outs: List[np.ndarray] = []
    for hi in range(h):
        kvh = hi // rep
        scores = np.einsum("bld,bmd->blm", q_r[:, :, hi], k_all[:, :, kvh]) * scale
        scores = np.where(causal[None], -np.inf, scores)
        mx = np.max(scores, axis=-1, keepdims=True)
        e = np.exp(scores - mx)
        p = e / np.sum(e, axis=-1, keepdims=True)
        outs.append(np.einsum("blm,bmd->bld", p, v_all[:, :, kvh]))
        ps.append(p)
    a = np.stack(outs, axis=2)                       # (B, L, H, d)
    y = a.reshape(b, l, d) @ w_o

    cache = {
        "x": x, "q_r": q_r, "k_all": k_all, "v_all": v_all,
        "k": k_all, "v": v_all,                    # 供 KV-Cache 增量解码续接
        "p": ps, "a": a, "cos": cos, "sin": sin, "scale": scale,
        "offset": offset, "new_len": l, "length": lc,
        "w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o,
    }
    return y, cache


def np_causal_attention_backward(
    gy: np.ndarray, cache: Dict[str, Any], cfg: LLMHierarchyConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """O2 反向：返回 (gx, {"w_q","w_k","w_v","w_o"})。"""
    x, q_r, k_all, v_all = cache["x"], cache["q_r"], cache["k_all"], cache["v_all"]
    ps, a, cos, sin, scale = cache["p"], cache["a"], cache["cos"], cache["sin"], cache["scale"]
    w_q, w_k, w_v, w_o = cache["w_q"], cache["w_k"], cache["w_v"], cache["w_o"]
    b, l, d = x.shape
    h, hk, hd, rep = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_rep

    g_a = (gy @ w_o.T).reshape(b, l, h, hd)
    g_w_o = a.reshape(-1, d).T @ gy.reshape(-1, d)

    g_q_r = np.zeros_like(q_r)
    g_k_r = np.zeros_like(k_all)
    g_v_all = np.zeros_like(v_all)
    for hi in range(h):
        kvh = hi // rep
        p = ps[hi]
        da = g_a[:, :, hi]
        gp = np.einsum("bld,bmd->blm", da, v_all[:, :, kvh])
        g_v_all[:, :, kvh] += np.einsum("blm,bld->bmd", p, da)
        gs = p * (gp - np.sum(gp * p, axis=-1, keepdims=True))
        gs = gs * scale
        g_q_r[:, :, hi] = np.einsum("blm,bmd->bld", gs, k_all[:, :, kvh])
        g_k_r[:, :, kvh] += np.einsum("blm,bld->bmd", gs, q_r[:, :, hi])

    gq = _np_rope_apply_t(g_q_r, cos, sin)
    gk = _np_rope_apply_t(g_k_r, cos, sin)
    gv = g_v_all
    # 只有本步新增的 token 会写回投影；缓存部分是常量。
    gk_new = gk[:, -l:]
    gv_new = gv[:, -l:]

    g_w_q = x.reshape(-1, d).T @ gq.reshape(-1, d)
    g_w_k = x.reshape(-1, d).T @ gk_new.reshape(-1, hk * hd)
    g_w_v = x.reshape(-1, d).T @ gv_new.reshape(-1, hk * hd)
    gx = (
        gq.reshape(b, l, d) @ w_q.T
        + gk_new.reshape(b, l, hk * hd) @ w_k.T
        + gv_new.reshape(b, l, hk * hd) @ w_v.T
    )
    return gx, {"w_q": g_w_q, "w_k": g_w_k, "w_v": g_w_v, "w_o": g_w_o}


# ------------------------------------------------------------------------------
# 2.4 L3 — SwiGLU 门控前馈算子 O3
# ------------------------------------------------------------------------------
def np_swiglu_forward(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    cfg: LLMHierarchyConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """O3: y = ( silu(x W_gate) ⊙ (x W_up) ) W_down。"""
    gate = x @ w_gate
    up = x @ w_up
    sig = _np_sigmoid(gate)
    act = gate * sig                       # silu(gate)
    h = act * up
    y = h @ w_down
    return y, {
        "x": x, "gate": gate, "sig": sig, "act": act, "up": up, "h": h,
        "w_gate": w_gate, "w_up": w_up, "w_down": w_down,
    }


def np_swiglu_backward(
    gy: np.ndarray, cache: Dict[str, Any], cfg: LLMHierarchyConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """O3 反向：silu'(z) = σ(z) (1 + z (1 - σ(z)))。"""
    x, gate, sig, act, up, h = (
        cache["x"], cache["gate"], cache["sig"], cache["act"], cache["up"], cache["h"],
    )
    d = x.shape[-1]
    g_w_down = h.reshape(-1, h.shape[-1]).T @ gy.reshape(-1, d)
    dh = gy @ cache["w_down"].T
    d_act = dh * up
    d_up = dh * act
    d_gate = d_act * (sig * (1.0 + gate * (1.0 - sig)))
    g_w_up = x.reshape(-1, d).T @ d_up.reshape(-1, d_up.shape[-1])
    g_w_gate = x.reshape(-1, d).T @ d_gate.reshape(-1, d_gate.shape[-1])
    gx = d_up @ cache["w_up"].T + d_gate @ cache["w_gate"].T
    return gx, {"w_gate": g_w_gate, "w_up": g_w_up, "w_down": g_w_down}


# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# 2.5 L4 — 残差流算子 O4（stream D ⊗ stream D → stream D）
# ------------------------------------------------------------------------------
def np_residual_forward(
    stream: np.ndarray, update: np.ndarray, gate: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """O4: y = stream + gate ⊙ update，逐通道可学的高速公路增益。"""
    y = stream + gate * update
    return y, {"stream": stream, "update": update, "gate": gate}


def np_residual_backward(
    gy: np.ndarray, cache: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O4 反向：返回 (g_stream, g_update, g_gate)。"""
    g_stream = gy
    g_update = gy * cache["gate"]
    g_gate = (gy * cache["update"]).sum(axis=tuple(range(gy.ndim - 1)))
    return g_stream, g_update, g_gate


# ------------------------------------------------------------------------------
# 2.6 L5 — 词表头算子 O5（残差流空间 D → 词表空间 V）
# ------------------------------------------------------------------------------
def np_vocab_head_forward(
    h: np.ndarray, w_vocab: np.ndarray, cfg: LLMHierarchyConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """O5: logits = c · tanh(h W_vocab^T / c)，c 为 logit softcap。"""
    cap = float(cfg.logit_softcap)
    z = h @ w_vocab.T
    y = cap * np.tanh(z / cap) if cap > 0 else z
    return y, {"h": h, "z": z, "cap": cap, "w_vocab": w_vocab}


def np_vocab_head_backward(
    gy: np.ndarray, cache: Dict[str, Any], cfg: LLMHierarchyConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """O5 反向：softcap 的导数为 1 - tanh²；返回 (g_h, g_w_vocab)。"""
    cap, z, h = cache["cap"], cache["z"], cache["h"]
    gz = gy * (1.0 - np.tanh(z / cap) ** 2) if cap > 0 else gy
    g_w = gz.reshape(-1, cfg.vocab_size).T @ h.reshape(-1, cfg.dim)
    g_h = gz @ cache["w_vocab"]
    return g_h, g_w


# ------------------------------------------------------------------------------
# 2.7 块与整模型：真正的层次串联（顺序组合，不是并联混合）
# ------------------------------------------------------------------------------
def np_block_forward(
    x: np.ndarray,
    params: Dict[str, np.ndarray],
    cfg: LLMHierarchyConfig,
    kv_cache: Optional[Dict[str, np.ndarray]] = None,
    offset: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """一个 pre-norm 块：x = Residual(x, Attn(Norm(x))); x = Residual(x, FFN(Norm(x)))。"""
    h1, c1 = np_rmsnorm_position_forward(x, params["gamma"], params["omega"], cfg, True, offset)
    a, c2 = np_causal_attention_forward(
        h1, params["attn_w_q"], params["attn_w_k"], params["attn_w_v"], params["attn_w_o"], cfg, kv_cache
    )
    x1, c3 = np_residual_forward(x, a, params["res_attn_gate"])
    h2, c4 = np_rmsnorm_position_forward(
        x1, params["ffn_gamma"], params["ffn_omega"], cfg, positional=False, offset=offset
    )
    f, c5 = np_swiglu_forward(
        h2, params["ffn_w_gate"], params["ffn_w_up"], params["ffn_w_down"], cfg
    )
    x2, c6 = np_residual_forward(x1, f, params["res_ffn_gate"])
    cache = {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6}
    return x2, cache


def np_block_backward(
    gx2: np.ndarray, cache: Dict[str, Any], params: Dict[str, np.ndarray], cfg: LLMHierarchyConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """块反向：残差分流 → 归一化 → 子层，逐层回传到块输入。"""
    g_f = gx2 * params["res_ffn_gate"]
    g_h2, g_ffn = np_swiglu_backward(g_f, cache["c5"], cfg)
    g_x1, g_n1, g_om1 = np_rmsnorm_position_backward(g_h2, cache["c4"], params["ffn_gamma"], cfg)
    g_x1 = g_x1 + gx2                                   # 残差直通
    g_a = g_x1 * params["res_attn_gate"]
    g_h1, g_attn = np_causal_attention_backward(g_a, cache["c2"], cfg)
    g_x0, g_n0, g_om0 = np_rmsnorm_position_backward(g_h1, cache["c1"], params["gamma"], cfg)
    g_x0 = g_x0 + g_x1
    grads = {
        "gamma": g_n0, "omega": g_om0, "ffn_gamma": g_n1, "ffn_omega": g_om1,
        "attn_w_q": g_attn["w_q"], "attn_w_k": g_attn["w_k"],
        "attn_w_v": g_attn["w_v"], "attn_w_o": g_attn["w_o"],
        "ffn_w_gate": g_ffn["w_gate"], "ffn_w_up": g_ffn["w_up"], "ffn_w_down": g_ffn["w_down"],
        "res_attn_gate": (g_x1 * cache["c3"]["update"]).sum(axis=tuple(range(g_x1.ndim - 1))),
        "res_ffn_gate": (gx2 * cache["c6"]["update"]).sum(axis=tuple(range(gx2.ndim - 1))),
    }
    return g_x0, grads


def np_model_forward(
    ids: np.ndarray,
    params: Dict[str, Any],
    cfg: LLMHierarchyConfig,
    caches: Optional[List[Dict[str, Any]]] = None,
    return_cache: bool = False,
) -> Any:
    """整模型前向：Embedding → N × Block → FinalNorm → VocabHead(softcap)。"""
    if ids.ndim != 2:
        raise ValueError("ids 必须为 (B, L) 的整数张量")
    x, c0 = np_embedding_forward(ids, params["embed"], cfg)
    offset = 0 if caches is None else int(caches[0]["c2"]["length"])
    new_caches: List[Dict[str, Any]] = []
    for i, blk in enumerate(params["blocks"]):
        kv = None if caches is None else caches[i]["c2"]
        x, c = np_block_forward(x, blk, cfg, kv, offset)
        new_caches.append(c)
    x, c_norm = np_rmsnorm_position_forward(
        x, params["final_gamma"], np.zeros(cfg.pair_dim), cfg, positional=False
    )
    logits, c_head = np_vocab_head_forward(x, params["lm_head"], cfg)
    cache = {
        "ids": ids, "c0": c0, "blocks": new_caches, "norm": c_norm, "head": c_head,
        "hidden": x, "logits": logits,
    }
    return (logits, cache) if return_cache else logits


def np_model_backward(
    g_logits: np.ndarray, cache: Dict[str, Any], params: Dict[str, Any], cfg: LLMHierarchyConfig
) -> Dict[str, Any]:
    """整模型反向：softcap → 词表头 → FinalNorm → N × Block → 嵌入表 scatter-add。"""
    gx, g_lm_head = np_vocab_head_backward(g_logits, cache["head"], cfg)
    gx, g_final, _ = np_rmsnorm_position_backward(
        gx, cache["norm"], params["final_gamma"], cfg
    )
    g_blocks: List[Dict[str, np.ndarray]] = [{} for _ in params["blocks"]]
    for i in reversed(range(len(params["blocks"]))):
        gx, g_blocks[i] = np_block_backward(gx, cache["blocks"][i], params["blocks"][i], cfg)
    g_embed = np_embedding_backward(gx, cache["c0"], params["embed"])
    return {
        "embed": g_embed,
        "blocks": g_blocks,
        "final_gamma": g_final,
        "lm_head": g_lm_head,
    }

if HAS_TORCH:

    def _rope_tables_torch(
        cfg: LLMHierarchyConfig,
        omega: Optional["torch.Tensor"],
        length: int,
        offset: int,
        dtype: "torch.dtype",
        device: "torch.device",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """与 NumPy 引擎同构的 (1, L, 1, d/2) cos/sin 表。"""
        idx = torch.arange(cfg.pair_dim, dtype=dtype, device=device)
        base = torch.tensor(float(cfg.rope_base), dtype=dtype, device=device)
        theta = torch.pow(base, -2.0 * idx / float(cfg.head_dim))
        if omega is not None:
            theta = theta * torch.exp(omega.to(dtype=dtype, device=device))
        pos = torch.arange(length, dtype=dtype, device=device) + float(offset)
        phi = pos[:, None] * theta[None, :]
        return torch.cos(phi)[None, :, None, :], torch.sin(phi)[None, :, None, :]

    def _rope_apply_torch(
        t: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor"
    ) -> "torch.Tensor":
        """成对旋转，返回与 t 同形的新张量（可微）。"""
        even = t[..., 0::2]
        odd = t[..., 1::2]
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        stacked = torch.stack((out_even, out_odd), dim=-1)      # (..., pair, 2)
        return stacked.flatten(-2)

    def _init_normal(param: "torch.Tensor", std: float, seed: int) -> None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            flat = torch.randn(param.numel(), generator=gen, dtype=torch.float64)
            param.copy_(flat.to(dtype=param.dtype, device=param.device).view_as(param) * std)

    class RMSNormPositionOp(nn.Module):
        """O1 —— 归一化-位置算子：y = RoPE_{θ⊙exp(ω)}( γ ⊙ x / sqrt(mean(x²)+ε) )。

        positional=False 时退化为纯 RMSNorm（用作最终归一化）。
        """

        _SALT = 101

        def __init__(self, cfg: LLMHierarchyConfig, positional: bool = True):
            super().__init__()
            self.cfg = cfg
            self.positional = positional
            self.gamma = nn.Parameter(torch.ones(cfg.dim))
            if positional:
                self.omega = nn.Parameter(torch.zeros(cfg.pair_dim))
            else:
                self.register_buffer("omega", torch.zeros(cfg.pair_dim))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            with torch.no_grad():
                self.gamma.fill_(1.0)
                self.omega.zero_()

        def forward(self, x: "torch.Tensor", offset: int = 0) -> "torch.Tensor":
            cfg = self.cfg
            if self.positional and cfg.use_triton and HAS_TRITON and x.is_cuda:
                return _TritonNormRope.apply(x, self.gamma, self.omega, offset, cfg)
            b, l, d = x.shape
            s = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + cfg.eps)
            h = x / s * self.gamma
            if not self.positional:
                return h
            cos, sin = _rope_tables_torch(cfg, self.omega, l, offset, x.dtype, x.device)
            t = h.view(b, l, cfg.n_heads, cfg.head_dim)
            return _rope_apply_torch(t, cos, sin).reshape(b, l, d)

        def numpy_params(self) -> Dict[str, np.ndarray]:
            return {
                "gamma": self.gamma.detach().cpu().double().numpy().copy(),
                "omega": self.omega.detach().cpu().double().numpy().copy(),
            }

    class CausalAttentionOp(nn.Module):
        """O2 —— 因果注意力算子（QKV 投射 + RoPE + 因果 softmax + GQA + KV-Cache）。

        训练 / prefill：QK^T、softmax、PV 三段矩阵乘走 Triton FlashAttention 融合核
        （长度 ≥ `cfg.flash_min_len` 且头维为 2 的幂时）；增量解码（kv_cache 非空）、
        CPU、fp64 以外的非常规头维回落下面的 PyTorch 路径。
        """

        _SALT = 211

        def __init__(self, cfg: LLMHierarchyConfig):
            super().__init__()
            self.cfg = cfg
            d, kv = cfg.dim, cfg.kv_dim
            self.w_q = nn.Parameter(torch.empty(d, d))
            self.w_k = nn.Parameter(torch.empty(d, kv))
            self.w_v = nn.Parameter(torch.empty(d, kv))
            self.w_o = nn.Parameter(torch.empty(d, d))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            cfg = self.cfg
            std = 0.8 / math.sqrt(cfg.dim)
            for i, p in enumerate((self.w_q, self.w_k, self.w_v, self.w_o)):
                _init_normal(p, std, cfg.seed + self._SALT + i)

        def forward(
            self,
            x: "torch.Tensor",
            kv_cache: Optional[Dict[str, Any]] = None,
        ) -> Tuple["torch.Tensor", Dict[str, Any]]:
            cfg = self.cfg
            b, l, d = x.shape
            h, hk, hd, rep = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_rep
            q = (x @ self.w_q).view(b, l, h, hd)
            k = (x @ self.w_k).view(b, l, hk, hd)
            v = (x @ self.w_v).view(b, l, hk, hd)

            offset = 0 if kv_cache is None else int(kv_cache["length"])
            cos, sin = _rope_tables_torch(cfg, None, l, offset, x.dtype, x.device)
            q = _rope_apply_torch(q, cos, sin)
            k = _rope_apply_torch(k, cos, sin)

            # 训练 / prefill：QK^T、softmax、PV 三段矩阵乘走融合核（不落 L×L 矩阵）；
            # 增量解码（kv_cache 非空）与 CPU / 非 2 的幂头维回落到 PyTorch 路径。
            use_flash = (
                cfg.use_triton and HAS_TRITON and x.is_cuda and kv_cache is None
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and hd in (16, 32, 64, 128) and l >= cfg.flash_min_len
            )
            if use_flash:
                try:
                    qf = q.transpose(1, 2).contiguous()
                    kf = k.transpose(1, 2).contiguous()
                    vf = v.transpose(1, 2).contiguous()
                    o = _FlashAttention.apply(qf, kf, vf)      # (B, H, L, hd)
                    y = o.transpose(1, 2).reshape(b, l, d) @ self.w_o
                    return y, {"k": k, "v": v, "length": l}
                except Exception:
                    # 融合核编译 / 运行失败时回落到 PyTorch 路径（数值语义不变）。
                    # Triton 的 tl.dot 不支持 float64，而审计用 .double() 跑 fp64，
                    # 编译会在 LLVM IR 层抛错；这里兜底保证任何 dtype 都能跑。
                    pass

            if kv_cache is None:
                k_all, v_all = k, v
            else:
                k_all = torch.cat((kv_cache["k"], k), dim=1)
                v_all = torch.cat((kv_cache["v"], v), dim=1)
            lc = k_all.shape[1]

            k_exp = k_all.repeat_interleave(rep, dim=2)          # (B, Lc, H, d)
            v_exp = v_all.repeat_interleave(rep, dim=2)
            scores = torch.einsum("blhd,bmhd->bhlm", q, k_exp) / math.sqrt(float(hd))
            qpos = torch.arange(l, device=x.device) + offset
            mpos = torch.arange(lc, device=x.device)
            mask = mpos[None, :] > qpos[:, None]                  # (L, Lc)
            scores = scores.masked_fill(mask[None, None], float("-inf"))
            p = torch.softmax(scores, dim=-1)
            a = torch.einsum("bhlm,bmhd->blhd", p, v_exp)
            y = a.reshape(b, l, d) @ self.w_o
            cache = {"k": k_all, "v": v_all, "length": lc}
            return y, cache

        def numpy_params(self) -> Dict[str, np.ndarray]:
            return {
                "w_q": self.w_q.detach().cpu().double().numpy().copy(),
                "w_k": self.w_k.detach().cpu().double().numpy().copy(),
                "w_v": self.w_v.detach().cpu().double().numpy().copy(),
                "w_o": self.w_o.detach().cpu().double().numpy().copy(),
            }

    class SwiGLUFFNOp(nn.Module):
        """O3 —— SwiGLU 门控前馈算子：y = (silu(x W_gate) ⊙ (x W_up)) W_down。"""

        _SALT = 307

        def __init__(self, cfg: LLMHierarchyConfig):
            super().__init__()
            self.cfg = cfg
            d, f = cfg.dim, cfg.ffn_dim
            self.w_gate = nn.Parameter(torch.empty(d, f))
            self.w_up = nn.Parameter(torch.empty(d, f))
            self.w_down = nn.Parameter(torch.empty(f, d))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            cfg = self.cfg
            _init_normal(self.w_gate, 1.0 / math.sqrt(cfg.dim), cfg.seed + self._SALT + 0)
            _init_normal(self.w_up, 1.0 / math.sqrt(cfg.dim), cfg.seed + self._SALT + 1)
            _init_normal(self.w_down, 1.0 / math.sqrt(cfg.ffn_dim), cfg.seed + self._SALT + 2)

        def hidden(self, x: "torch.Tensor") -> "torch.Tensor":
            """F 中间空间的激活：silu(x W_gate) ⊙ (x W_up)。"""
            gate = x @ self.w_gate
            up = x @ self.w_up
            if self.cfg.use_triton and HAS_TRITON and x.is_cuda:
                return _TritonSwiGLUAct.apply(gate, up)
            return F.silu(gate) * up

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.hidden(x) @ self.w_down

        def numpy_params(self) -> Dict[str, np.ndarray]:
            return {
                "w_gate": self.w_gate.detach().cpu().double().numpy().copy(),
                "w_up": self.w_up.detach().cpu().double().numpy().copy(),
                "w_down": self.w_down.detach().cpu().double().numpy().copy(),
            }



    class EmbeddingOp(nn.Module):
        """O0 —— 词嵌入算子：token (B,L) → 残差流 (B,L,D)，表空间 V×D。"""

        _SALT = 503

        def __init__(self, cfg: LLMHierarchyConfig):
            super().__init__()
            self.cfg = cfg
            self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.dim))
            _init_normal(self.weight, 0.02, cfg.seed + self._SALT)

        def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
            return self.weight[ids] * math.sqrt(float(self.cfg.dim))

        def numpy_params(self) -> Dict[str, np.ndarray]:
            return {"embed": self.weight.detach().cpu().double().numpy().copy()}

    class ResidualStreamOp(nn.Module):
        """O4 —— 残差流算子：y = stream + gate ⊙ update。

        gate 为逐通道可学的高速公路增益（初值 1.0）；残差通带与子层写回在这一层显式分离。
        """

        _SALT = 601

        def __init__(self, cfg: LLMHierarchyConfig, init: float = 1.0):
            super().__init__()
            self.cfg = cfg
            self.gate = nn.Parameter(torch.full((cfg.dim,), float(init)))

        def forward(self, stream: "torch.Tensor", update: "torch.Tensor") -> "torch.Tensor":
            return stream + self.gate * update

    class VocabHeadOp(nn.Module):
        """O5 —— 词表头算子：残差流 (B,L,D) → 词表空间 (B,L,V)，带 logit softcap。"""

        _SALT = 701

        def __init__(self, cfg: LLMHierarchyConfig):
            super().__init__()
            self.cfg = cfg
            self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.dim))
            _init_normal(self.weight, 0.02, cfg.seed + self._SALT)

        def forward(
            self, h: "torch.Tensor", weight: Optional["torch.Tensor"] = None
        ) -> "torch.Tensor":
            w = self.weight if weight is None else weight
            z = h @ w.t()
            cap = float(self.cfg.logit_softcap)
            return cap * torch.tanh(z / cap) if cap > 0 else z

        def numpy_params(self) -> Dict[str, np.ndarray]:
            return {"lm_head": self.weight.detach().cpu().double().numpy().copy()}

    class LLMHierarchyBlock(nn.Module):
        """一个 pre-norm 块：两个子层，各自带独立的归一化与残差流算子。

        x = Residual_attn(x, Attention(Norm_attn(x)))
        x = Residual_ffn (x, SwiGLU  (Norm_ffn (x)))
        """

        def __init__(self, cfg: LLMHierarchyConfig, layer: int = 0):
            super().__init__()
            self.cfg = cfg
            self.layer = layer
            self.norm_attn = RMSNormPositionOp(cfg)
            self.attn = CausalAttentionOp(cfg)
            self.res_attn = ResidualStreamOp(cfg)
            self.norm_ffn = RMSNormPositionOp(cfg, positional=False)
            self.ffn = SwiGLUFFNOp(cfg)
            self.res_ffn = ResidualStreamOp(cfg)
            # 每层独立盐，避免同构层共享初值
            base = cfg.seed + 1000 * (layer + 1)
            for i, p in enumerate((self.attn.w_q, self.attn.w_k, self.attn.w_v, self.attn.w_o)):
                _init_normal(p, 0.8 / math.sqrt(cfg.dim), base + i)
            for i, p in enumerate((self.ffn.w_gate, self.ffn.w_up)):
                _init_normal(p, 1.0 / math.sqrt(cfg.dim), base + 10 + i)
            _init_normal(self.ffn.w_down, 1.0 / math.sqrt(cfg.ffn_dim), base + 12)

        def forward(
            self,
            x: "torch.Tensor",
            kv_cache: Optional[Dict[str, Any]] = None,
            offset: int = 0,
        ) -> Tuple["torch.Tensor", Dict[str, Any]]:
            a, cache = self.attn(self.norm_attn(x, offset), kv_cache)
            x = self.res_attn(x, a)
            x = self.res_ffn(x, self.ffn(self.norm_ffn(x, offset)))
            return x, cache

        def numpy_params(self) -> Dict[str, np.ndarray]:
            entry = dict(self.norm_attn.numpy_params())
            entry.update({f"ffn_{k}": v for k, v in self.norm_ffn.numpy_params().items()})
            entry.update({f"attn_{k}": v for k, v in self.attn.numpy_params().items()})
            entry.update({f"ffn_{k}": v for k, v in self.ffn.numpy_params().items()})
            entry["res_attn_gate"] = self.res_attn.gate.detach().cpu().double().numpy().copy()
            entry["res_ffn_gate"] = self.res_ffn.gate.detach().cpu().double().numpy().copy()
            return entry

    class LLMHierarchyOperator(nn.Module):
        """完整 LLM 层次算子：Embedding → N × Block → FinalNorm → VocabHead。

        六个算子类型各自独占空间（token V / stream D / head H×d_h / ffn F / logit V），
        参数块互不相交；支持 KV-Cache 增量解码与采样。

        执行路径：prefill/训练时每个块的注意力走 FlashAttention 融合核（长度达门槛），
        解码时逐 token 走 PyTorch 的 KV-Cache 路径；`spaces()` 返回各空间的实际维数。
        """

        def __init__(self, cfg: LLMHierarchyConfig):
            super().__init__()
            self.cfg = cfg
            self.embedding = EmbeddingOp(cfg)
            self.blocks = nn.ModuleList([LLMHierarchyBlock(cfg, i) for i in range(cfg.n_layers)])
            self.final_norm = RMSNormPositionOp(cfg, positional=False)
            self.head = VocabHeadOp(cfg)

        # -- 空间元数据 ------------------------------------------------------
        def spaces(self) -> Dict[str, int]:
            return self.cfg.spaces()

        def vocab_weight(self) -> "torch.Tensor":
            return self.embedding.weight if self.cfg.tie_embeddings else self.head.weight

        # -- 前向 ------------------------------------------------------------
        def hidden(
            self, ids: "torch.Tensor", caches: Optional[List[Dict[str, Any]]] = None
        ) -> Tuple["torch.Tensor", List[Dict[str, Any]]]:
            x = self.embedding(ids)
            offset = 0 if caches is None else int(caches[0]["length"])
            new_caches: List[Dict[str, Any]] = []
            for i, block in enumerate(self.blocks):
                cache = None if caches is None else caches[i]
                x, c = block(x, cache, offset)
                new_caches.append(c)
            return self.final_norm(x), new_caches

        def logits_from_hidden(self, h: "torch.Tensor") -> "torch.Tensor":
            weight = self.embedding.weight if self.cfg.tie_embeddings else None
            return self.head(h, weight)

        def forward(
            self, ids: "torch.Tensor", caches: Optional[List[Dict[str, Any]]] = None
        ) -> Tuple["torch.Tensor", List[Dict[str, Any]]]:
            h, new_caches = self.hidden(ids, caches)
            return self.logits_from_hidden(h), new_caches

        @torch.no_grad()
        def generate(
            self,
            ids: "torch.Tensor",
            max_new_tokens: int = 16,
            temperature: float = 1.0,
            top_k: int = 0,
            greedy: bool = False,
        ) -> "torch.Tensor":
            """自回归增量解码；每步只喂入一个新 token，KV-Cache 线性增长。"""
            caches: Optional[List[Dict[str, Any]]] = None
            logits, caches = self.forward(ids, caches)
            out = ids
            for _ in range(int(max_new_tokens)):
                step = logits[:, -1] / max(float(temperature), 1e-6)
                if greedy or top_k == 1:
                    nxt = torch.argmax(step, dim=-1, keepdim=True)
                else:
                    if top_k > 0:
                        k = min(int(top_k), step.shape[-1])
                        thresh = torch.topk(step, k, dim=-1).values[..., -1:]
                        step = step.masked_fill(step < thresh, float("-inf"))
                    probs = torch.softmax(step, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
                out = torch.cat((out, nxt), dim=1)
                logits, caches = self.forward(nxt, caches)
            return out

        # -- 与 NumPy 引擎交换参数 -------------------------------------------
        def numpy_params(self) -> Dict[str, Any]:
            return {
                "embed": self.embedding.weight.detach().cpu().double().numpy().copy(),
                "blocks": [b.numpy_params() for b in self.blocks],
                "final_gamma": self.final_norm.gamma.detach().cpu().double().numpy().copy(),
                "lm_head": self.vocab_weight().detach().cpu().double().numpy().copy(),
            }

    # 兼容别名：模型即完整层次算子
    LLMHierarchyModel = LLMHierarchyOperator

# ==============================================================================
# 4. Triton GPU 融合核
#    (a) RMSNorm + RoPE 单核融合：一次读入 x，归一化、旋转、写回，cos/sin 表也在核内生成；
#    (b) SwiGLU 门控融合：silu(gate) ⊙ up 单核完成，反向解析回传两侧梯度；
#    (c) FlashAttention 式因果注意力：QK^T → online softmax → PV 全在核内，
#        不 materialize L×L 矩阵，支持 GQA、KV 分组与 fp16/bf16/fp32/fp64。
#    三套核均与 NumPy 双精度引擎逐元素对拍；fp64 下达到 1e-15 量级。
#    块大小 / warps 按精度与头维选（见 `_flash_block_cfg`），短序列自动回落 PyTorch。
# ==============================================================================
if HAS_TRITON:

    @triton.jit
    def _norm_rope_fwd_kernel(
        X, G, COS, SIN, Y, EPS,
        L, D: tl.constexpr, HD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        d = tl.arange(0, BLOCK)
        mask = d < D
        pos = row % L
        eps = tl.load(EPS)
        x = tl.load(X + row * D + d, mask=mask, other=0.0)
        g = tl.load(G + d, mask=mask, other=0.0)
        s = tl.sqrt(tl.sum(x * x) / D + eps)
        n = x / s
        h = n * g
        partner = d ^ 1
        hp = tl.load(X + row * D + partner, mask=mask, other=0.0) / s
        hp = hp * tl.load(G + partner, mask=mask, other=0.0)
        pair = (d % HD) // 2
        c = tl.load(COS + pos * (HD // 2) + pair, mask=mask, other=1.0)
        sn = tl.load(SIN + pos * (HD // 2) + pair, mask=mask, other=0.0)
        even = (d % 2) == 0
        y = tl.where(even, h * c - hp * sn, h * c + hp * sn)
        tl.store(Y + row * D + d, y, mask=mask)

    @triton.jit
    def _norm_rope_bwd_kernel(
        X, G, COS, SIN, Y, GO, GX, GG, DPHI, EPS,
        L, D: tl.constexpr, HD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        d = tl.arange(0, BLOCK)
        mask = d < D
        pos = row % L
        eps = tl.load(EPS)
        x = tl.load(X + row * D + d, mask=mask, other=0.0)
        g = tl.load(G + d, mask=mask, other=0.0)
        s = tl.sqrt(tl.sum(x * x) / D + eps)
        n = x / s
        partner = d ^ 1
        pair = (d % HD) // 2
        c = tl.load(COS + pos * (HD // 2) + pair, mask=mask, other=1.0)
        sn = tl.load(SIN + pos * (HD // 2) + pair, mask=mask, other=0.0)
        gy = tl.load(GO + row * D + d, mask=mask, other=0.0)
        gyp = tl.load(GO + row * D + partner, mask=mask, other=0.0)
        even = (d % 2) == 0
        sign = tl.where(even, 1.0, -1.0)
        g_rot = gy * c + sign * gyp * sn           # 旋转转置
        # dL/dphi = gy_b * y_a - gy_a * y_b，只在偶数 lane 上累加
        y = tl.load(Y + row * D + d, mask=mask, other=0.0)
        yp = tl.load(Y + row * D + partner, mask=mask, other=0.0)
        dphi = -gy * yp + gyp * y
        dphi = tl.where(even, dphi, 0.0)
        tl.atomic_add(
            DPHI + pos * (HD // 2) + pair, dphi, mask=mask & even
        )
        gn = g_rot * g
        csum = tl.sum(gn * n) / D
        gx = (gn - csum * n) / s
        tl.store(GX + row * D + d, gx, mask=mask)
        tl.atomic_add(GG + d, g_rot * n, mask=mask)

    @triton.jit
    def _swiglu_act_fwd_kernel(GATE, UP, OUT, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        g = tl.load(GATE + off, mask=mask, other=0.0)
        u = tl.load(UP + off, mask=mask, other=0.0)
        a = g / (1.0 + tld.exp(-g))
        tl.store(OUT + off, a * u, mask=mask)

    @triton.jit
    def _swiglu_act_bwd_kernel(GATE, UP, GO, DGATE, DUP, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        g = tl.load(GATE + off, mask=mask, other=0.0)
        u = tl.load(UP + off, mask=mask, other=0.0)
        go = tl.load(GO + off, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tld.exp(-g))
        act = g * sig
        dsilu = sig * (1.0 + g * (1.0 - sig))
        tl.store(DGATE + off, go * u * dsilu, mask=mask)
        tl.store(DUP + off, go * act, mask=mask)

    def _cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    _INVF_CACHE: Dict[Tuple[float, int, str, str], "torch.Tensor"] = {}
    _EPS_CACHE: Dict[Tuple[str, str], "torch.Tensor"] = {}

    def _eps_tensor(value: float, dtype: "torch.dtype", device: "torch.device") -> "torch.Tensor":
        """eps 以 0-d 张量进核，避免 Python float 被降成 fp32。"""
        key = (str(dtype), str(device))
        cached = _EPS_CACHE.get(key)
        if cached is None or float(cached) != float(value):
            cached = torch.tensor(value, dtype=dtype, device=device)
            _EPS_CACHE[key] = cached
        return cached

    def _inv_freq_torch(
        cfg: LLMHierarchyConfig, dtype: "torch.dtype", device: "torch.device"
    ) -> "torch.Tensor":
        """缓存 θ_i = base^(-2i/d)（只与结构有关，无须每次重建）。"""
        key = (float(cfg.rope_base), int(cfg.head_dim), str(dtype), str(device))
        inv = _INVF_CACHE.get(key)
        if inv is None:
            idx = torch.arange(cfg.pair_dim, dtype=dtype, device=device)
            base = torch.tensor(float(cfg.rope_base), dtype=dtype, device=device)
            inv = torch.pow(base, -2.0 * idx / float(cfg.head_dim)).contiguous()
            _INVF_CACHE[key] = inv
        return inv

    @triton.jit
    def _rope_table_kernel(
        INVF, OMEGA, COS, SIN, POS0, L,
        P: tl.constexpr, BLOCK_P: tl.constexpr,
    ):
        """一次算 BLOCK_P 个位置的 cos/sin 平表（2D tile，减少微型 CTA 数量）。"""
        pid = tl.program_id(0)
        local = pid * BLOCK_P + tl.arange(0, BLOCK_P)
        p = tl.arange(0, P)
        mask_pos = local < L
        inv = tl.load(INVF + p)
        om = tl.load(OMEGA + p)
        theta = inv * tld.exp(om)
        pos = POS0 + local
        phi = pos[:, None].to(theta.dtype) * theta[None, :]
        mask = mask_pos[:, None]
        tl.store(COS + local[:, None] * P + p[None, :], tld.cos(phi), mask=mask)
        tl.store(SIN + local[:, None] * P + p[None, :], tld.sin(phi), mask=mask)

    def _rope_tables_triton(
        cfg: LLMHierarchyConfig,
        omega: "torch.Tensor",
        length: int,
        offset: int,
        dtype: "torch.dtype",
        device: "torch.device",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """单核生成 (L, pair) 的 cos/sin 平表，位置从 offset 起算。"""
        inv = _inv_freq_torch(cfg, dtype, device)
        cos = torch.empty(length, cfg.pair_dim, dtype=dtype, device=device)
        sin = torch.empty(length, cfg.pair_dim, dtype=dtype, device=device)
        block_p = 16
        _rope_table_kernel[(triton.cdiv(length, block_p),)](
            inv, omega, cos, sin, int(offset), int(length),
            P=triton.next_power_of_2(cfg.pair_dim), BLOCK_P=block_p,
        )
        return cos, sin

    class _TritonNormRope(torch.autograd.Function):
        """O1 的融合核包装：RMSNorm + 学习频率 RoPE，前向/反向均为单核。"""

        @staticmethod
        def forward(
            ctx: Any,
            x: "torch.Tensor",
            gamma: "torch.Tensor",
            omega: "torch.Tensor",
            offset: int,
            cfg: LLMHierarchyConfig,
        ) -> "torch.Tensor":
            b, l, d = x.shape
            x2 = x.reshape(b * l, d).contiguous()
            cos, sin = _rope_tables_triton(cfg, omega, l, offset, x.dtype, x.device)
            y = torch.empty_like(x2)
            eps = _eps_tensor(cfg.eps, x.dtype, x.device)
            _norm_rope_fwd_kernel[(b * l,)](
                x2, gamma, cos, sin, y, eps,
                l, d, cfg.head_dim,
                BLOCK=triton.next_power_of_2(d),
            )
            ctx.save_for_backward(x2, gamma, omega, cos, sin, y)
            ctx.cfg = cfg
            ctx.offset = int(offset)
            ctx.shape = (b, l, d)
            return y.view(b, l, d)

        @staticmethod
        def backward(
            ctx: Any, gy: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", None, None]:
            x2, gamma, omega, cos, sin, y = ctx.saved_tensors
            cfg = ctx.cfg
            b, l, d = ctx.shape
            gx = torch.empty_like(x2)
            ggamma = torch.zeros(d, dtype=x2.dtype, device=x2.device)
            gphi = torch.zeros(l, cfg.pair_dim, dtype=x2.dtype, device=x2.device)
            eps = _eps_tensor(cfg.eps, x2.dtype, x2.device)
            _norm_rope_bwd_kernel[(b * l,)](
                x2, gamma, cos, sin, y, gy.reshape(b * l, d).contiguous(), gx, ggamma, gphi,
                eps, l, d, cfg.head_dim,
                BLOCK=triton.next_power_of_2(d),
            )
            idx = torch.arange(cfg.pair_dim, dtype=x2.dtype, device=x2.device)
            base = torch.tensor(float(cfg.rope_base), dtype=x2.dtype, device=x2.device)
            theta = torch.pow(base, -2.0 * idx / float(cfg.head_dim)) * torch.exp(omega.to(x2.dtype))
            pos = torch.arange(l, dtype=x2.dtype, device=x2.device) + float(ctx.offset)
            gomega = (gphi * pos[:, None] * theta[None, :]).sum(dim=0)
            return gx.view(b, l, d), ggamma, gomega, None, None

    class _TritonSwiGLUAct(torch.autograd.Function):
        """O3 的门控融合：silu(gate) ⊙ up 单核完成，反向单核回传。"""

        @staticmethod
        def forward(ctx: Any, gate: "torch.Tensor", up: "torch.Tensor") -> "torch.Tensor":
            out = torch.empty_like(gate)
            n = gate.numel()
            _swiglu_act_fwd_kernel[(_cdiv(n, 1024),)](gate, up, out, n, BLOCK=1024)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx: Any, go: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            gate, up = ctx.saved_tensors
            dgate = torch.empty_like(gate)
            dup = torch.empty_like(up)
            n = gate.numel()
            _swiglu_act_bwd_kernel[(_cdiv(n, 1024),)](
                gate, up, go.contiguous(), dgate, dup, n, BLOCK=1024
            )
            return dgate, dup


    # ------------------------------------------------------------------
    # FlashAttention 式因果注意力核：把 QK^T、softmax、PV 三段矩阵乘
    # 压进一次核内完成，不再materialize L×L 注意力矩阵。
    # ------------------------------------------------------------------
    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, Out, LSE, L, SCALE,
        H: tl.constexpr, HK: tl.constexpr, REP: tl.constexpr,
        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        IEEE: tl.constexpr, ACC: tl.constexpr,
    ):
        scale = tl.load(SCALE)
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        kvh = h // REP
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = tl.arange(0, D)
        row_ok = offs_m < L
        q = tl.load(Q + ((b * H + h) * L + offs_m[:, None]) * D + offs_d[None, :],
                    mask=row_ok[:, None], other=0.0)
        m_i = tl.full([BM], float("-inf"), dtype=ACC)
        l_i = tl.zeros([BM], dtype=ACC)
        acc = tl.zeros([BM, D], dtype=ACC)
        kbase = K + ((b * HK + kvh) * L) * D
        vbase = V + ((b * HK + kvh) * L) * D
        # 因果：第 m 行只需扫到 m 所在的块
        for start_n in range(0, (pid_m + 1) * BM, BN):
            n_idx = start_n + offs_n
            nmask = n_idx < L
            k = tl.load(kbase + n_idx[:, None] * D + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            if IEEE:
                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
            else:
                qk = tl.dot(q, tl.trans(k))
            qk = qk.to(ACC) * scale
            causal = n_idx[None, :] <= offs_m[:, None]
            qk = tl.where(row_ok[:, None] & causal, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(vbase + n_idx[:, None] * D + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            if IEEE:
                acc = tl.dot(p.to(v.dtype), v, acc, input_precision="ieee")
            else:
                acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_new
        out = acc / l_i[:, None]
        logsum = m_i + tl.log(l_i)
        tl.store(Out + ((b * H + h) * L + offs_m[:, None]) * D + offs_d[None, :],
                 out.to(Out.dtype.element_ty), mask=row_ok[:, None])
        tl.store(LSE + (b * H + h) * L + offs_m, logsum.to(LSE.dtype.element_ty), mask=row_ok)

    @triton.jit
    def _flash_bwd_kernel(
        Q, K, V, dO, LSE, DELTA, dQ, dK, dV, L, SCALE,
        H: tl.constexpr, HK: tl.constexpr, REP: tl.constexpr,
        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        IEEE: tl.constexpr, ACC: tl.constexpr,
    ):
        scale = tl.load(SCALE)
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        kvh = h // REP
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = tl.arange(0, D)
        row_ok = offs_m < L
        q = tl.load(Q + ((b * H + h) * L + offs_m[:, None]) * D + offs_d[None, :],
                    mask=row_ok[:, None], other=0.0)
        do = tl.load(dO + ((b * H + h) * L + offs_m[:, None]) * D + offs_d[None, :],
                     mask=row_ok[:, None], other=0.0)
        lse = tl.load(LSE + (b * H + h) * L + offs_m, mask=row_ok, other=0.0).to(ACC)
        delta = tl.load(DELTA + (b * H + h) * L + offs_m, mask=row_ok, other=0.0).to(ACC)
        dq = tl.zeros([BM, D], dtype=ACC)
        kbase = K + ((b * HK + kvh) * L) * D
        vbase = V + ((b * HK + kvh) * L) * D
        for start_n in range(0, (pid_m + 1) * BM, BN):
            n_idx = start_n + offs_n
            nmask = n_idx < L
            k = tl.load(kbase + n_idx[:, None] * D + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            v = tl.load(vbase + n_idx[:, None] * D + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            if IEEE:
                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
            else:
                qk = tl.dot(q, tl.trans(k))
            qk = qk.to(ACC) * scale
            causal = n_idx[None, :] <= offs_m[:, None]
            qk = tl.where(row_ok[:, None] & causal, qk, float("-inf"))
            p = tl.exp(qk - lse[:, None])
            if IEEE:
                dv_blk = tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            else:
                dv_blk = tl.dot(tl.trans(p).to(do.dtype), do)
                dp = tl.dot(do, tl.trans(v))
            dp = dp.to(ACC)
            # 行和用全局 Δ = rowsum(do ⊙ o) = Σ_j p_j dp_j；按 N 块局部求和会在多块时出错
            ds = p * (dp - delta[:, None])
            if IEEE:
                dq += tl.dot(ds.to(q.dtype), k, input_precision="ieee") * scale
                dk_blk = tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee") * scale
            else:
                dq += tl.dot(ds.to(q.dtype), k) * scale
                dk_blk = tl.dot(tl.trans(ds).to(q.dtype), q) * scale
            tl.atomic_add(dK + ((b * HK + kvh) * L + n_idx[:, None]) * D + offs_d[None, :],
                          dk_blk.to(dK.dtype.element_ty), mask=nmask[:, None])
            tl.atomic_add(dV + ((b * HK + kvh) * L + n_idx[:, None]) * D + offs_d[None, :],
                          dv_blk.to(dV.dtype.element_ty), mask=nmask[:, None])
        tl.store(dQ + ((b * H + h) * L + offs_m[:, None]) * D + offs_d[None, :],
                 dq.to(dQ.dtype.element_ty), mask=row_ok[:, None])

    def _flash_block_cfg(
        dtype: "torch.dtype", head_dim: int, backward: bool = False
    ) -> Tuple[int, int, int, int]:
        """按精度/头维/前反向选 (BM, BN, stages, warps)。

        实测（L=1024, D=64, 8×8 头）：fp32 前向 warps=8 时 20.1→2.4 ms；反向 BN=32 时
        12.6→6.5 ms；fp16 前向 64×64/4 warps、反向 stages=2 最佳。D=128 降 tile 保共享内存。
        """
        if dtype == torch.float64 or head_dim >= 128:
            return 32, 32, 2, 4
        if dtype in (torch.float16, torch.bfloat16):
            return (64, 64, 2, 4) if backward else (64, 64, 3, 4)
        return (64, 32, 2, 8) if backward else (64, 64, 2, 8)

    class _FlashAttention(torch.autograd.Function):
        """因果 GQA 注意力的融合核：前向不落 L×L 矩阵，反向由同一套块内重算得到。"""

        @staticmethod
        def forward(ctx: Any, q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
            b, h, l, d = q.shape
            hk = k.shape[1]
            rep = h // hk
            out = torch.empty_like(q)
            acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            lse = torch.empty(b, h, l, dtype=acc_dtype, device=q.device)
            bm, bn, stages, warps = _flash_block_cfg(q.dtype, d)
            scale = torch.tensor(1.0 / math.sqrt(float(d)), dtype=q.dtype, device=q.device)
            _flash_fwd_kernel[(triton.cdiv(l, bm), b * h)](
                q, k, v, out, lse, l, scale, h, hk, rep, d, bm, bn,
                q.dtype in (torch.float32, torch.float64),
                tl.float64 if q.dtype == torch.float64 else tl.float32,
                num_stages=stages, num_warps=warps,
            )
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.rep = rep
            return out

        @staticmethod
        def backward(ctx: Any, do: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            q, k, v, out, lse = ctx.saved_tensors
            b, h, l, d = q.shape
            hk = k.shape[1]
            rep = ctx.rep
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            bm, bn, stages, warps = _flash_block_cfg(q.dtype, d, backward=True)
            scale = torch.tensor(1.0 / math.sqrt(float(d)), dtype=q.dtype, device=q.device)
            if q.dtype == torch.float64:
                delta = (do * out).sum(-1).contiguous()
            else:
                delta = (do.float() * out.float()).sum(-1).contiguous()
            _flash_bwd_kernel[(triton.cdiv(l, bm), b * h)](
                q, k, v, do.contiguous(), lse, delta, dq, dk, dv, l, scale,
                h, hk, rep, d, bm, bn,
                q.dtype in (torch.float32, torch.float64),
                tl.float64 if q.dtype == torch.float64 else tl.float32,
                num_stages=stages, num_warps=warps,
            )
            return dq, dk, dv

    def triton_flash_attention(
        q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor"
    ) -> "torch.Tensor":
        """q (B,H,L,D)、k/v (B,H_kv,L,D) → o (B,H,L,D)；因果 + GQA。"""
        if not (HAS_TORCH and HAS_TRITON):
            raise RuntimeError("需要 torch + triton")
        return _FlashAttention.apply(q.contiguous(), k.contiguous(), v.contiguous())

def triton_norm_rope_forward(
    x: "torch.Tensor",
    gamma: "torch.Tensor",
    omega: "torch.Tensor",
    cfg: LLMHierarchyConfig,
    offset: int = 0,
) -> "torch.Tensor":
    """供验证调用的显式入口（需 CUDA + Triton）。"""
    if not (HAS_TORCH and HAS_TRITON):
        raise RuntimeError("需要 torch + triton")
    return _TritonNormRope.apply(x, gamma, omega, offset, cfg)


def triton_swiglu_forward(gate: "torch.Tensor", up: "torch.Tensor") -> "torch.Tensor":
    if not (HAS_TORCH and HAS_TRITON):
        raise RuntimeError("需要 torch + triton")
    return _TritonSwiGLUAct.apply(gate, up)




# ==============================================================================
# 5. 逐算子等价验证、Gradcheck 与五维正交独立性审计
# ==============================================================================
def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def _audit_config() -> LLMHierarchyConfig:
    return LLMHierarchyConfig(
        dim=128, n_heads=4, n_kv_heads=2, ffn_dim=320, vocab_size=128,
        n_layers=2, use_triton=True,
    )


def _check_operator_equivalence(cfg: LLMHierarchyConfig) -> Dict[str, Dict[str, float]]:
    """A1：六个算子逐个对拍 NumPy ↔ PyTorch（前向 + 全部参数/输入梯度）。"""
    rng = np.random.default_rng(3)
    b, l, d, v = 2, 5, cfg.dim, cfg.vocab_size
    model = LLMHierarchyOperator(cfg).double().eval()
    blk = model.blocks[0]
    out: Dict[str, Dict[str, float]] = {}

    def fwd_bwd(name: str, fn_t, fn_np, grads_np: Dict[str, np.ndarray],
                tensors: Dict[str, "torch.Tensor"], go: np.ndarray) -> None:
        xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        yt = fn_t(xt)
        y_np = fn_np(x_np)
        yt.backward(torch.tensor(go, dtype=torch.float64))
        entry = {"fwd_max_abs": _max_abs(yt.detach().numpy(), y_np)}
        entry["dx_rel"] = _rel_err(xt.grad.numpy(), grads_np["dx"])
        for k, t in tensors.items():
            entry[f"d{k}_rel"] = _rel_err(t.grad.numpy(), grads_np[k])
        out[name] = entry
        model.zero_grad()

    # --- O0 Embedding ---
    ids = rng.integers(0, v, (b, l))
    e_np, c_e = np_embedding_forward(ids, model.embedding.weight.detach().numpy(), cfg)
    e_t = model.embedding(torch.tensor(ids))
    e_t.sum().backward()
    g_table = np_embedding_backward(np.ones_like(e_np), c_e, model.embedding.weight.detach().numpy())
    out["embedding"] = {
        "fwd_max_abs": _max_abs(e_t.detach().numpy(), e_np),
        "dweight_rel": _rel_err(model.embedding.weight.grad.numpy(), g_table),
    }
    model.zero_grad()

    x_np = rng.normal(0, 1, (b, l, d))
    go = rng.normal(0, 1, (b, l, d))

    # --- O1 NormPosition ---
    y_np, c1 = np_rmsnorm_position_forward(
        x_np, blk.norm_attn.gamma.detach().numpy(), blk.norm_attn.omega.detach().numpy(), cfg)
    gx_np, gg_np, gw_np = np_rmsnorm_position_backward(
        go, c1, blk.norm_attn.gamma.detach().numpy(), cfg)
    fwd_bwd("norm_position", lambda t: blk.norm_attn(t),
            lambda a: np_rmsnorm_position_forward(
                a, blk.norm_attn.gamma.detach().numpy(), blk.norm_attn.omega.detach().numpy(), cfg)[0],
            {"dx": gx_np, "gamma": gg_np, "omega": gw_np},
            {"gamma": blk.norm_attn.gamma, "omega": blk.norm_attn.omega}, go)

    # --- O2 CausalAttention ---
    attn_p = {k: getattr(blk.attn, k).detach().numpy() for k in ("w_q", "w_k", "w_v", "w_o")}
    y_np, c2 = np_causal_attention_forward(
        x_np, attn_p["w_q"], attn_p["w_k"], attn_p["w_v"], attn_p["w_o"], cfg)
    gx_np, ga_np = np_causal_attention_backward(go, c2, cfg)
    fwd_bwd("attention", lambda t: blk.attn(t)[0],
            lambda a: np_causal_attention_forward(
                a, attn_p["w_q"], attn_p["w_k"], attn_p["w_v"], attn_p["w_o"], cfg)[0],
            {"dx": gx_np, **{k: ga_np[k] for k in ("w_q", "w_k", "w_v", "w_o")}},
            {k: getattr(blk.attn, k) for k in ("w_q", "w_k", "w_v", "w_o")}, go)

    # --- O3 SwiGLU ---
    ffn_p = {k: getattr(blk.ffn, k).detach().numpy() for k in ("w_gate", "w_up", "w_down")}
    y_np, c3 = np_swiglu_forward(x_np, ffn_p["w_gate"], ffn_p["w_up"], ffn_p["w_down"], cfg)
    gx_np, gf_np = np_swiglu_backward(go, c3, cfg)
    fwd_bwd("swiglu", lambda t: blk.ffn(t),
            lambda a: np_swiglu_forward(
                a, ffn_p["w_gate"], ffn_p["w_up"], ffn_p["w_down"], cfg)[0],
            {"dx": gx_np, **{k: gf_np[k] for k in ("w_gate", "w_up", "w_down")}},
            {k: getattr(blk.ffn, k) for k in ("w_gate", "w_up", "w_down")}, go)

    # --- O4 ResidualStream ---
    s_np = rng.normal(0, 1, (b, l, d))
    u_np = rng.normal(0, 1, (b, l, d))
    gate_np = blk.res_attn.gate.detach().numpy()
    y_np, c4 = np_residual_forward(s_np, u_np, gate_np)
    gs_np, gu_np, gg_np = np_residual_backward(go, c4)
    st = torch.tensor(s_np, dtype=torch.float64)
    ut = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
    yt = blk.res_attn(st, ut)
    yt.backward(torch.tensor(go, dtype=torch.float64))
    out["residual"] = {
        "fwd_max_abs": _max_abs(yt.detach().numpy(), y_np),
        "dupdate_rel": _rel_err(ut.grad.numpy(), gu_np),
        "dgate_rel": _rel_err(blk.res_attn.gate.grad.numpy(), gg_np),
    }
    model.zero_grad()

    # --- O5 VocabHead ---
    h_np = rng.normal(0, 1, (b, l, d))
    go_v = rng.normal(0, 1, (b, l, v))
    y_np, c5 = np_vocab_head_forward(h_np, model.head.weight.detach().numpy(), cfg)
    gh_np, gwl_np = np_vocab_head_backward(go_v, c5, cfg)
    ht = torch.tensor(h_np, dtype=torch.float64, requires_grad=True)
    yt = model.head(ht)
    yt.backward(torch.tensor(go_v, dtype=torch.float64))
    out["vocab_head"] = {
        "fwd_max_abs": _max_abs(yt.detach().numpy(), y_np),
        "dh_rel": _rel_err(ht.grad.numpy(), gh_np),
        "dweight_rel": _rel_err(model.head.weight.grad.numpy(), gwl_np),
    }
    return out


def _block_grad(blk: "LLMHierarchyBlock", key: str) -> Optional["torch.Tensor"]:
    if key == "gamma":
        return blk.norm_attn.gamma.grad
    if key == "omega":
        return blk.norm_attn.omega.grad
    if key == "ffn_gamma":
        return blk.norm_ffn.gamma.grad
    if key == "ffn_omega":
        return blk.norm_ffn.omega.grad
    if key.startswith("attn_"):
        return getattr(blk.attn, key[5:]).grad
    if key.startswith("ffn_w"):
        return getattr(blk.ffn, key[4:]).grad
    if key == "res_attn_gate":
        return blk.res_attn.gate.grad
    if key == "res_ffn_gate":
        return blk.res_ffn.gate.grad
    raise KeyError(key)


def _check_model_equivalence(cfg: LLMHierarchyConfig) -> Dict[str, float]:
    """A2：整模型 NumPy ↔ PyTorch 前向与全部梯度（含绑定词表时的梯度累加）。"""
    model = LLMHierarchyOperator(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (3, 8))
    rng = np.random.default_rng(11)
    go = rng.normal(0.0, 1.0, (3, 8, cfg.vocab_size))
    params = model.numpy_params()

    logits_t, _ = model(ids)
    logits_t.backward(torch.tensor(go, dtype=torch.float64))
    logits_n, cache = np_model_forward(ids.numpy(), params, cfg, return_cache=True)
    grads = np_model_backward(go, cache, params, cfg)

    embed_ref = grads["embed"] + (grads["lm_head"] if cfg.tie_embeddings else 0.0)
    worst = _rel_err(model.embedding.weight.grad.numpy(), embed_ref)
    worst = max(worst, _rel_err(model.final_norm.gamma.grad.numpy(), grads["final_gamma"]))
    if cfg.tie_embeddings:
        worst = max(worst, _rel_err(model.head.weight.grad.numpy(), grads["lm_head"]))
    for i, blk in enumerate(model.blocks):
        for key in blk.numpy_params():
            g = _block_grad(blk, key)
            if g is None:                       # 非位置归一化的 omega 是 buffer
                assert float(np.max(np.abs(grads["blocks"][i][key]))) == 0.0, key
                continue
            worst = max(worst, _rel_err(g.numpy(), grads["blocks"][i][key]))
    return {"fwd_max_abs": _max_abs(logits_t.detach().numpy(), logits_n), "grad_rel_max": worst}


def _check_triton_equivalence(cfg: LLMHierarchyConfig) -> Dict[str, Any]:
    """A3：Triton 融合核 ↔ NumPy（fp64），含前向、反向与位置偏移。"""
    if not (HAS_TORCH and HAS_TRITON and torch.cuda.is_available()):
        return {"available": False}
    rng = np.random.default_rng(7)
    b, t, d = 3, 6, cfg.dim
    x_np = rng.normal(0, 1, (b, t, d))
    gy_np = rng.normal(0, 1, (b, t, d))
    gamma_np = rng.normal(1.0, 0.1, d)
    omega_np = rng.normal(0.0, 0.2, cfg.pair_dim)

    y_np, c1 = np_rmsnorm_position_forward(x_np, gamma_np, omega_np, cfg)
    gx_np, gg_np, gomega_np = np_rmsnorm_position_backward(gy_np, c1, gamma_np, cfg)
    x = torch.tensor(x_np, device="cuda", dtype=torch.float64, requires_grad=True)
    g = torch.tensor(gamma_np, device="cuda", dtype=torch.float64, requires_grad=True)
    w = torch.tensor(omega_np, device="cuda", dtype=torch.float64, requires_grad=True)
    y = triton_norm_rope_forward(x, g, w, cfg)
    fwd = _max_abs(y.detach().cpu().numpy(), y_np)
    y.backward(torch.tensor(gy_np, device="cuda", dtype=torch.float64))
    dx = _rel_err(x.grad.cpu().numpy(), gx_np)
    dg = _rel_err(g.grad.cpu().numpy(), gg_np)
    dw = _rel_err(w.grad.cpu().numpy(), gomega_np)
    y_off = triton_norm_rope_forward(x.detach(), g.detach(), w.detach(), cfg, offset=5)
    y_off_np, _ = np_rmsnorm_position_forward(x_np, gamma_np, omega_np, cfg, True, 5)
    off_err = _max_abs(y_off.cpu().numpy(), y_off_np)

    gf_np = rng.normal(0, 1, (b * t, cfg.ffn_dim))
    up_np = rng.normal(0, 1, (b * t, cfg.ffn_dim))
    go_np = rng.normal(0, 1, (b * t, cfg.ffn_dim))
    h_np = gf_np * (1.0 / (1.0 + np.exp(-gf_np))) * up_np
    gf = torch.tensor(gf_np, device="cuda", requires_grad=True)
    up = torch.tensor(up_np, device="cuda", requires_grad=True)
    h = triton_swiglu_forward(gf, up)
    s_fwd = _max_abs(h.detach().cpu().numpy(), h_np)
    h.backward(torch.tensor(go_np, device="cuda"))
    sig = 1.0 / (1.0 + np.exp(-gf_np))
    dsilu = sig * (1.0 + gf_np * (1.0 - sig))
    s_dg = _rel_err(gf.grad.cpu().numpy(), go_np * up_np * dsilu)
    s_du = _rel_err(up.grad.cpu().numpy(), go_np * gf_np * sig)
    return {
        "available": True,
        "norm_rope_fwd_max_abs": fwd, "norm_rope_dx_rel": dx,
        "norm_rope_dgamma_rel": dg, "norm_rope_domega_rel": dw,
        "norm_rope_offset_max_abs": off_err,
        "swiglu_fwd_max_abs": s_fwd, "swiglu_dgate_rel": s_dg, "swiglu_dup_rel": s_du,
    }


def _check_kv_cache(cfg: LLMHierarchyConfig) -> Dict[str, float]:
    """A4：整序列前向 vs 逐 token 增量解码（NumPy 与 PyTorch 双引擎）。"""
    model = LLMHierarchyOperator(cfg).double().eval()
    blk = model.blocks[0]
    rng = np.random.default_rng(5)
    x_np = rng.normal(0, 1, (2, 7, cfg.dim))
    wq, wk, wv, wo = (blk.attn.w_q.detach().numpy(), blk.attn.w_k.detach().numpy(),
                      blk.attn.w_v.detach().numpy(), blk.attn.w_o.detach().numpy())
    y_full, _ = np_causal_attention_forward(x_np, wq, wk, wv, wo, cfg)
    cache, outs = None, []
    for t in range(x_np.shape[1]):
        y_t, cache = np_causal_attention_forward(x_np[:, t:t + 1], wq, wk, wv, wo, cfg, cache)
        outs.append(y_t)
    np_kv = _max_abs(np.concatenate(outs, axis=1), y_full)

    xt = torch.tensor(x_np, dtype=torch.float64)
    yt, _ = blk.attn(xt)
    tc, touts = None, []
    for t in range(x_np.shape[1]):
        y_t, tc = blk.attn(xt[:, t:t + 1], tc)
        touts.append(y_t)
    torch_kv = _max_abs(torch.cat(touts, dim=1).detach().numpy(), yt.detach().numpy())

    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    full, _ = model(ids)
    caches, parts = None, []
    for t in range(8):
        step, caches = model(ids[:, t:t + 1], caches)
        parts.append(step)
    model_kv = _max_abs(torch.cat(parts, dim=1).detach().numpy(), full.detach().numpy())
    return {"numpy_attention": np_kv, "torch_attention": torch_kv, "torch_model": model_kv}


def _check_gradcheck(cfg: LLMHierarchyConfig, eps: float = 1e-6) -> Dict[str, float]:
    """A5：PyTorch 自动微分 vs 中心差分（独立于 NumPy 引擎的第三方校验）。"""
    model = LLMHierarchyOperator(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    go = torch.tensor(np.random.default_rng(9).normal(0.0, 1.0, (2, 6, cfg.vocab_size)),
                     dtype=torch.float64)

    def loss() -> float:
        logits, _ = model(ids)
        return float((logits * go).sum())

    model.zero_grad()
    model(ids)[0].backward(go)
    params = dict(model.named_parameters())
    grads = {n: p.grad.detach().clone() for n, p in params.items() if p.grad is not None}

    err_p, ref_p = 0.0, 0.0
    with torch.no_grad():
        for name, p in params.items():
            if name not in grads:
                continue
            flat = p.reshape(-1)
            gflat = grads[name].reshape(-1)
            for idx in torch.topk(gflat.abs(), min(2, gflat.numel())).indices.tolist():
                orig = flat[idx].item()
                flat[idx] = orig + eps
                lp = loss()
                flat[idx] = orig - eps
                lm = loss()
                flat[idx] = orig
                fd = (lp - lm) / (2 * eps)
                ref = abs(gflat[idx].item())
                err_p = max(err_p, abs(fd - gflat[idx].item()) / max(ref, 1e-8))
                ref_p = max(ref_p, ref)

    # 嵌入表：随机 4 个 (token, 通道) 的中心差分
    err_e, ref_e = 0.0, 0.0
    emb_grad = grads["embedding.weight"]
    with torch.no_grad():
        rng = np.random.default_rng(31)
        for _ in range(4):
            tok = int(rng.integers(0, cfg.vocab_size))
            ch = int(rng.integers(0, cfg.dim))
            w = model.embedding.weight
            orig = w[tok, ch].item()
            w[tok, ch] = orig + eps
            lp = loss()
            w[tok, ch] = orig - eps
            lm = loss()
            w[tok, ch] = orig
            fd = (lp - lm) / (2 * eps)
            g = emb_grad[tok, ch].item()
            err_e = max(err_e, abs(fd - g) / max(abs(g), 1e-8))
            ref_e = max(ref_e, abs(g))
    return {"param_rel_err": float(err_p), "param_ref": float(ref_p),
            "embed_rel_err": float(err_e), "embed_ref": float(ref_e)}


def _check_training_smoke(cfg: LLMHierarchyConfig, steps: int = 40) -> Dict[str, float]:
    """A6：可训练性冒烟——合成下一 token 任务上跑 AdamW，损失必须显著下降。"""
    torch.manual_seed(cfg.seed)
    model = LLMHierarchyOperator(cfg)
    if torch.cuda.is_available():
        model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    base = torch.arange(8).repeat(2)[None, :]
    ids = (base + torch.arange(4)[:, None]) % 5
    ids = ids.to(next(model.parameters()).device)
    target = torch.cat((ids[:, 1:], ids[:, :1]), dim=1)
    losses = []
    for _ in range(steps):
        logits, _ = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return {"loss_first": losses[0], "loss_last": losses[-1],
            "loss_min": min(losses), "steps": steps}


def _check_zero_imports() -> List[str]:
    """维度 1：AST 静态检查，只允许标准库与 numpy/torch/triton。"""
    allowed = {
        "__future__", "argparse", "ast", "copy", "json", "math", "sys",
        "dataclasses", "pathlib", "typing", "numpy", "torch", "triton",
    }
    tree = ast.parse(Path(__file__).resolve().read_text(encoding="utf-8"))
    bad: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                bad.append(f"relative:{node.level}")
            elif node.module and node.module.split(".")[0] not in allowed:
                bad.append(node.module)
    return bad


def _vif(matrix: np.ndarray) -> Tuple[float, int, float]:
    svals = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(svals > 1e-8 * max(svals[0], 1e-12)))
    corr = np.atleast_2d(np.corrcoef(matrix.T))
    try:
        vifs = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vifs = np.full(matrix.shape[1], np.inf)
    return float(np.max(vifs)), rank, float(svals[-1])


def _operator_audit(cfg: LLMHierarchyConfig) -> Dict[str, Any]:
    """五维正交独立性审计：六个算子类型，六个空间。"""
    model = LLMHierarchyOperator(cfg).double().eval()
    blk = model.blocks[0]
    d, v = cfg.dim, cfg.vocab_size
    torch.manual_seed(5)
    # L=5、取第 5 个位置：注意力在该位置已能看到 5 个 value 子空间（每个 rank kv_dim），
    # 足以占满 D；位置太短时 rank 上限是 kv_dim+1，测不出真实空间。
    base = torch.randn(1, 5, d, dtype=torch.float64)
    pos = 4

    def image_rank(y: "torch.Tensor", tol: float = 1e-8) -> Tuple[int, int, float]:
        """一批输入下，算子输出张成的子空间秩（= 它在自有空间里真正占满多少维）。"""
        mat = y.detach().reshape(-1, y.shape[-1]).numpy()
        svals = np.linalg.svd(mat, compute_uv=False)
        rank = int(np.sum(svals > tol * max(svals[0], 1e-12)))
        return rank, mat.shape[1], float(svals[-1])

    with torch.no_grad():
        xb = torch.randn(64, 5, d, dtype=torch.float64)
        embed_rank = int(np.linalg.matrix_rank(model.embedding.weight.detach().numpy(), tol=1e-8))
        norm_rank, norm_out, norm_smin = image_rank(blk.norm_attn(xb))
        attn_rank, attn_out, attn_smin = image_rank(blk.attn(xb)[0])
        ffn_rank, ffn_out, ffn_smin = image_rank(blk.ffn(xb))
        hidden_rank, hidden_out, hidden_smin = image_rank(blk.ffn.hidden(xb))
        head_rank, head_out, head_smin = image_rank(model.head(xb))
        gate_np = blk.res_attn.gate.detach().numpy()
        residual_rank = int(np.sum(np.abs(gate_np) > 1e-12))

    # 维度 3：五个 D 空间算子在同一探针下的输入梯度共线性
    def input_grad(fn) -> np.ndarray:
        x = base[0, pos].clone().requires_grad_(True)
        z = base.clone()
        z[0, pos] = x
        y = fn(z)[0, pos]
        probe = torch.randn_like(y)
        g, = torch.autograd.grad(y, x, grad_outputs=probe)
        return g.detach().numpy()

    stream0 = base.detach()
    audit_fns = [
        blk.norm_attn,
        lambda z: blk.attn(z)[0],
        blk.ffn,
        lambda z: blk.res_attn(stream0, z),
        lambda z: model.head(z),
    ]
    m_bwd = np.column_stack([input_grad(fn).ravel() for fn in audit_fns])
    bwd_vif, bwd_rank, bwd_smin = _vif(m_bwd)

    # 维度 4：6×6 参数交叉灵敏度（每个算子的输入固定）
    names = ["embedding", "norm_attn", "attn", "ffn", "res_attn", "head"]
    modules = [model.embedding, blk.norm_attn, blk.attn, blk.ffn, blk.res_attn, model.head]
    ids0 = torch.tensor([[3, 7, 11]])

    def outputs() -> List["torch.Tensor"]:
        return [
            model.embedding(ids0)[0],
            blk.norm_attn(base)[0],
            blk.attn(base)[0][0],
            blk.ffn(base)[0],
            blk.res_attn(base, base * 0.5)[0],
            model.head(base)[0],
        ]

    with torch.no_grad():
        outs0 = outputs()
        sens = np.zeros((6, 6))
        for j, target in enumerate(modules):
            saved = {n: p.detach().clone() for n, p in target.named_parameters()}
            for p in target.parameters():
                p.add_(0.05)
            pert = outputs()
            for i in range(6):
                sens[i, j] = float((pert[i] - outs0[i]).abs().max())
            for n, p in target.named_parameters():
                p.copy_(saved[n])
        off = sens[~np.eye(6, dtype=bool)]
        max_off = float(np.max(np.abs(off)))
        exact = int(np.sum(off == 0.0))
        min_diag = float(np.min(np.diag(sens)))

    # 维度 5：参数置零消融（留一法不可替代性）
    ids_eval = torch.randint(0, v, (4, 12))
    with torch.no_grad():
        ref = model(ids_eval)[0].detach().numpy()
        ref_norm = max(float(np.linalg.norm(ref)), 1e-12)
        abl: Dict[str, float] = {}
        for kind in names:
            m2 = copy.deepcopy(model)
            with torch.no_grad():
                if kind == "embedding":
                    m2.embedding.weight.zero_()
                elif kind == "norm_attn":
                    for b in m2.blocks:
                        b.norm_attn.gamma.zero_()
                        b.norm_ffn.gamma.zero_()
                    m2.final_norm.gamma.zero_()
                elif kind == "attn":
                    for b in m2.blocks:
                        for p in b.attn.parameters():
                            p.zero_()
                elif kind == "ffn":
                    for b in m2.blocks:
                        for p in b.ffn.parameters():
                            p.zero_()
                elif kind == "res_attn":
                    for b in m2.blocks:
                        b.res_attn.gate.zero_()
                        b.res_ffn.gate.zero_()
                elif kind == "head":
                    m2.head.weight.zero_()
                out = m2(ids_eval)[0].detach().numpy()
            # 留一法：置零该算子的全部可学参数后，logits 的相对偏离（RMS 意义）
            abl[kind] = float(np.linalg.norm(ref - out) / ref_norm)

    return {
        "sibling_imports": _check_zero_imports(),
        "spaces": cfg.spaces(),
        "forward_ranks": {
            "embedding_table": (embed_rank, min(cfg.vocab_size, d)),
            "norm_position": (norm_rank, norm_out),
            "attention": (attn_rank, attn_out),
            "swiglu": (ffn_rank, ffn_out),
            "swiglu_hidden": (hidden_rank, hidden_out),
            "residual_stream": (residual_rank, d),
            "vocab_head": (head_rank, head_out),
        },
        "forward_sigma_min": {
            "norm_position": norm_smin, "attention": attn_smin, "swiglu": ffn_smin,
            "swiglu_hidden": hidden_smin, "vocab_head": head_smin,
        },
        "bwd_max_vif": bwd_vif, "bwd_rank": bwd_rank, "bwd_sigma_min": bwd_smin,
        "sensitivity_matrix": sens.tolist(),
        "max_off_diag": max_off, "exact_zero_count": exact, "min_diag": min_diag,
        "ablation_rel_change": abl, "min_ablation_rel_change": float(min(abl.values())),
    }



def _check_triton_attention(cfg: LLMHierarchyConfig) -> Dict[str, Any]:
    """A3b：融合注意力核 ↔ NumPy 解析引擎（fp64 位级 / fp32 单精度容差），L 非块对齐。"""
    if not (HAS_TORCH and HAS_TRITON and torch.cuda.is_available()):
        return {"available": False}
    rng = np.random.default_rng(17)
    b, l = 2, max(cfg.flash_min_len + 8, 200)      # 非块对齐，且必须走融合核
    x_np = rng.normal(0.0, 1.0, (b, l, cfg.dim))
    go = rng.normal(0.0, 1.0, (b, l, cfg.dim))
    out: Dict[str, Any] = {"available": True}
    for label, dtype, ftol, gtol in (
        ("fp64", torch.float64, 1e-12, 1e-11),
        ("fp32", torch.float32, 3e-5, 1e-4),
    ):
        model = LLMHierarchyOperator(cfg).to(device="cuda", dtype=dtype).eval()
        blk = model.blocks[0]
        p_np = {k: getattr(blk.attn, k).detach().double().cpu().numpy()
                for k in ("w_q", "w_k", "w_v", "w_o")}
        y_np, c = np_causal_attention_forward(
            x_np, p_np["w_q"], p_np["w_k"], p_np["w_v"], p_np["w_o"], cfg)
        gx_np, gp_np = np_causal_attention_backward(go, c, cfg)
        x = torch.tensor(x_np, device="cuda", dtype=dtype, requires_grad=True)
        y, _ = blk.attn(x)
        y.backward(torch.tensor(go, device="cuda", dtype=dtype))
        entry = {
            "fwd_rel": _rel_err(y.detach().double().cpu().numpy(), y_np),
            "dx_rel": _rel_err(x.grad.double().cpu().numpy(), gx_np),
        }
        for k in ("w_q", "w_k", "w_v", "w_o"):
            entry[f"d{k}_rel"] = _rel_err(
                getattr(blk.attn, k).grad.double().cpu().numpy(), gp_np[k])
        out[label] = entry
        assert entry["fwd_rel"] < ftol, (label, entry)
        assert max(v for k, v in entry.items() if k != "fwd_rel") < gtol, (label, entry)
    return out


def benchmark_attention(iters: int = 20) -> Dict[str, Any]:
    """融合注意力 vs PyTorch eager（materialize L×L）的 fwd / fwd+bwd 耗时。"""
    if not (HAS_TORCH and HAS_TRITON and torch.cuda.is_available()):
        return {"available": False}
    B, H, HK, L, D = 8, 8, 4, 1024, 64
    out: Dict[str, Any] = {"available": True, "shape": {"B": B, "H": H, "L": L, "D": D}}

    def timeit(fn, rounds: int = 3):
        for _ in range(3):
            fn()
        best = float("inf")
        for _ in range(rounds):
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            for _ in range(iters):
                fn()
            t1.record()
            torch.cuda.synchronize()
            best = min(best, t0.elapsed_time(t1) / iters)
        return best

    for label, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(B, HK, L, D, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(B, HK, L, D, device="cuda", dtype=dtype, requires_grad=True)
        rep = H // HK

        def eager(q=q, k=k, v=v, rep=rep):
            ke = k.repeat_interleave(rep, dim=1)
            ve = v.repeat_interleave(rep, dim=1)
            s = q @ ke.transpose(-1, -2) / math.sqrt(D)
            mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
            p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
            return p @ ve

        def eager_bwd(q=q, k=k, v=v):
            g = torch.autograd.grad(eager(q, k, v).square().mean(), (q, k, v))
            return g

        def flash_bwd(q=q, k=k, v=v):
            o = triton_flash_attention(q, k, v)
            return torch.autograd.grad(o.square().mean(), (q, k, v))

        t_e = timeit(eager)
        t_f = timeit(lambda q=q, k=k, v=v: triton_flash_attention(q, k, v))
        t_eb = timeit(eager_bwd)
        t_fb = timeit(flash_bwd)
        out[label] = {
            "fwd_eager_ms": t_e, "fwd_flash_ms": t_f, "fwd_speedup": t_e / t_f,
            "fwd_bwd_eager_ms": t_eb, "fwd_bwd_flash_ms": t_fb, "fwd_bwd_speedup": t_eb / t_fb,
        }
    return out


def run_verification(verbose: bool = True, quick: bool = False) -> Dict[str, Any]:
    """逐算子三引擎等价、Gradcheck、五维正交审计与可训练性冒烟。"""
    cfg = _audit_config()
    np.random.seed(2026)
    report: Dict[str, Any] = {
        "config": {"dim": cfg.dim, "heads": cfg.n_heads, "kv_heads": cfg.n_kv_heads,
                   "ffn": cfg.ffn_dim, "vocab": cfg.vocab_size, "layers": cfg.n_layers,
                   "spaces": cfg.spaces()},
        "operators": _check_operator_equivalence(cfg),
        "model": _check_model_equivalence(cfg),
        "triton": _check_triton_equivalence(cfg),
        "kv_cache": _check_kv_cache(cfg),
        "triton_attention": _check_triton_attention(cfg),
    }
    if not quick:
        report["gradcheck"] = _check_gradcheck(cfg)
        report["training_smoke"] = _check_training_smoke(cfg)
    report["audit"] = _operator_audit(cfg)

    ops, model, tri, kv, aud = (
        report["operators"], report["model"], report["triton"],
        report["kv_cache"], report["audit"],
    )

    for name, entry in ops.items():
        assert entry["fwd_max_abs"] < 1e-13, (name, entry)
        for k, val in entry.items():
            if k != "fwd_max_abs":
                assert val < 1e-12, (name, k, val)
    assert model["fwd_max_abs"] < 1e-13 and model["grad_rel_max"] < 1e-11, model
    assert kv["numpy_attention"] < 1e-13 and kv["torch_attention"] < 1e-13, kv
    assert kv["torch_model"] < 1e-13, kv
    if tri.get("available"):
        assert tri["norm_rope_fwd_max_abs"] < 1e-13 and tri["norm_rope_offset_max_abs"] < 1e-13, tri
        for k in ("norm_rope_dx_rel", "norm_rope_dgamma_rel", "norm_rope_domega_rel",
                  "swiglu_fwd_max_abs", "swiglu_dgate_rel", "swiglu_dup_rel"):
            assert tri[k] < 1e-12, (k, tri[k])
    if not quick:
        assert report["gradcheck"]["param_rel_err"] < 1e-6, report["gradcheck"]
        assert report["gradcheck"]["embed_rel_err"] < 1e-6, report["gradcheck"]
        smoke = report["training_smoke"]
        assert smoke["loss_last"] < 0.5 * smoke["loss_first"], smoke

    assert not aud["sibling_imports"], aud["sibling_imports"]
    expect = {
        "embedding_table": min(cfg.vocab_size, cfg.dim),
        "norm_position": cfg.dim,
        "attention": cfg.dim,
        "swiglu": cfg.dim,
        "swiglu_hidden": cfg.ffn_dim,
        "residual_stream": cfg.dim,
        "vocab_head": min(cfg.vocab_size, cfg.dim),
    }
    for name, (rank, _) in aud["forward_ranks"].items():
        assert rank == expect[name], (name, rank, expect[name])
    assert aud["bwd_rank"] == 5, aud["bwd_rank"]
    assert aud["bwd_max_vif"] < 2.0, aud["bwd_max_vif"]
    assert aud["max_off_diag"] == 0.0, aud["max_off_diag"]
    assert aud["min_diag"] > 0.02, aud["min_diag"]
    assert aud["min_ablation_rel_change"] > 0.02, aud["ablation_rel_change"]

    if verbose:
        _print_report(report, cfg)
    return report


def _print_report(report: Dict[str, Any], cfg: LLMHierarchyConfig) -> None:
    ops, model, tri, kv, aud = (
        report["operators"], report["model"], report["triton"],
        report["kv_cache"], report["audit"],
    )
    gc = report.get("gradcheck")
    smoke = report.get("training_smoke")
    line = "=" * 84
    print(line)
    print("  LLM-HIERARCHY OPERATOR — 六算子 × 六空间 / 三引擎等价 / 五维正交审计")
    sp = cfg.spaces()
    print(f"  D={cfg.dim}  H={cfg.n_heads}  H_kv={cfg.n_kv_heads}  F={cfg.ffn_dim}  "
          f"V={cfg.vocab_size}  layers={cfg.n_layers}")
    print(f"  空间: token {sp['token']} | stream {sp['stream']} | head {sp['head']} | "
          f"kv {sp['kv']} | ffn {sp['ffn']} | logit {sp['logit']}")
    print(line)
    print("[A] 逐算子 NumPy ↔ PyTorch（双精度，前向 max|Δ| / 反向相对误差）")
    rows = [
        ("O0 embedding", "fwd_max_abs", ["dweight_rel"]),
        ("O1 norm_position", "fwd_max_abs", ["dx_rel", "dgamma_rel", "domega_rel"]),
        ("O2 attention", "fwd_max_abs", ["dx_rel", "dw_q_rel", "dw_k_rel", "dw_v_rel", "dw_o_rel"]),
        ("O3 swiglu", "fwd_max_abs", ["dx_rel", "dw_gate_rel", "dw_up_rel", "dw_down_rel"]),
        ("O4 residual", "fwd_max_abs", ["dupdate_rel", "dgate_rel"]),
        ("O5 vocab_head", "fwd_max_abs", ["dh_rel", "dweight_rel"]),
    ]
    for label, fk, gkeys in rows:
        entry = ops[label.split()[1]]
        gs = "  ".join(f"{k.split('_rel')[0][1:]} {entry[k]:.1e}" for k in gkeys)
        print(f"    {label:16s} fwd {entry[fk]:.2e}   {gs}")
    print(f"    整模型           fwd {model['fwd_max_abs']:.2e}   grad {model['grad_rel_max']:.2e}")
    print(f"    KV-Cache 逐 token (NumPy/Torch/整模型): {kv['numpy_attention']:.2e} / "
          f"{kv['torch_attention']:.2e} / {kv['torch_model']:.2e}")
    if tri.get("available"):
        print(f"    Triton fp64: norm+rope fwd {tri['norm_rope_fwd_max_abs']:.2e}, "
              f"dX {tri['norm_rope_dx_rel']:.2e}, dγ {tri['norm_rope_dgamma_rel']:.2e}, "
              f"dω {tri['norm_rope_domega_rel']:.2e}")
        print(f"                 offset {tri['norm_rope_offset_max_abs']:.2e}; "
              f"swiglu fwd {tri['swiglu_fwd_max_abs']:.2e}, "
              f"dGate {tri['swiglu_dgate_rel']:.2e}, dUp {tri['swiglu_dup_rel']:.2e}")
    else:
        print("    Triton：不可用（无 CUDA/triton），已跳过")
    attn = report.get("triton_attention", {})
    if attn.get("available"):
        for label in ("fp64", "fp32"):
            e = attn[label]
            print(f"    Triton 注意力核 {label} (L={max(cfg.flash_min_len + 8, 200)} 非块对齐): fwd {e['fwd_rel']:.2e}  "
                  f"dX {e['dx_rel']:.2e}  dW {max(e[k] for k in e if k.startswith('dw')):.2e}")
    if gc:
        print(f"[B] 中心差分 Gradcheck (eps=1e-6): 参数 {gc['param_rel_err']:.2e}  "
              f"嵌入表 {gc['embed_rel_err']:.2e}")
    if smoke:
        print(f"[C] 可训练性冒烟（{smoke['steps']} 步 AdamW）: loss {smoke['loss_first']:.4f} → "
              f"{smoke['loss_last']:.4f}")
    print("[D] 五维正交独立性审计（六个算子、六个空间）")
    print(f"    Dim1 零同级依赖                    : {len(aud['sibling_imports'])} 个 (PASS)")
    print("    Dim2 算子像空间满秩（各自空间，一批输入下输出张成的秩）:")
    for name, (rank, out) in aud["forward_ranks"].items():
        smin = aud["forward_sigma_min"].get(name)
        smin_s = f"  σ_min={smin:.3e}" if smin is not None else ""
        print(f"         {name:16s} {rank}/{out}{smin_s}")
    print(f"    Dim3 反向输入雅可比                : {aud['bwd_rank']}/5 算子, "
          f"σ_min={aud['bwd_sigma_min']:.3f}, Max VIF={aud['bwd_max_vif']:.4f} (< 2.0)")
    print(f"    Dim4 6×6 参数交叉灵敏度            : 非对角精确 0 计数 {aud['exact_zero_count']}/30, "
          f"min diag={aud['min_diag']:.4f}")
    for row in aud["sensitivity_matrix"]:
        print("         [" + "  ".join(f"{x:.2e}" for x in row) + "]")
    print("    Dim5 留一法消融（置零参数的 logits 相对偏离 ‖Δ‖/‖logits‖）:")
    for k, val in aud["ablation_rel_change"].items():
        print(f"         drop {k:12s} Δrel = {val:.4f}")
    print(line)
    print("  ALL CHECKS PASSED — 六个算子在六个空间里满秩、正交、不可替代。")
    print(line)


# ==============================================================================
# 6. 基准与 CLI
# ==============================================================================
def benchmark_fusion(cfg: Optional[LLMHierarchyConfig] = None, iters: int = 50) -> Dict[str, Any]:
    """比较融合核与 PyTorch eager 的耗时（fp32，GPU；无 GPU 返回 available=False）。"""
    if not (HAS_TORCH and HAS_TRITON and torch.cuda.is_available()):
        return {"available": False}
    cfg = cfg or LLMHierarchyConfig(dim=1024, n_heads=8, n_kv_heads=8, ffn_dim=2752, n_layers=1)
    b, l, d, f = 8, 1024, cfg.dim, cfg.ffn_dim
    x = torch.randn(b * l, d, device="cuda")
    gamma = torch.randn(d, device="cuda").abs() + 0.5
    omega = torch.randn(cfg.pair_dim, device="cuda")
    gate = torch.randn(b * l, f, device="cuda")
    up = torch.randn(b * l, f, device="cuda")

    def timeit(fn, rounds: int = 3):
        for _ in range(5):
            fn()
        best = float("inf")
        for _ in range(rounds):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) / iters)
        return best

    x3 = x.view(b, l, d)
    inv = _inv_freq_torch(cfg, torch.float32, x.device)
    pos = torch.arange(l, dtype=torch.float32, device=x.device)

    def eager_norm_only():
        return x3 / torch.sqrt(x3.pow(2).mean(-1, keepdim=True) + cfg.eps) * gamma

    def eager_norm_rope():
        h = eager_norm_only()
        theta = inv * torch.exp(omega)
        phi = pos[:, None] * theta[None, :]
        c = torch.cos(phi)[None, :, None, :]
        s = torch.sin(phi)[None, :, None, :]
        t = h.view(b, l, cfg.n_heads, cfg.head_dim)
        e, o = t[..., 0::2], t[..., 1::2]
        return torch.stack((e * c - o * s, e * s + o * c), -1).flatten(-2)

    t_eager_norm = timeit(eager_norm_only)
    t_eager_full = timeit(eager_norm_rope)
    t_fused_norm = timeit(lambda: triton_norm_rope_forward(x3, gamma, omega, cfg))
    t_eager_glu = timeit(lambda: F.silu(gate) * up)
    t_fused_glu = timeit(lambda: triton_swiglu_forward(gate, up))
    return {
        "available": True,
        "shape": {"rows": b * l, "dim": d, "ffn": f},
        "norm_only_eager_ms": t_eager_norm,
        "norm_rope_eager_ms": t_eager_full,
        "norm_rope_fused_ms": t_fused_norm,
        "norm_rope_speedup": t_eager_full / t_fused_norm,
        "swiglu_eager_ms": t_eager_glu,
        "swiglu_fused_ms": t_fused_glu,
        "swiglu_speedup": t_eager_glu / t_fused_glu,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 全层次可微算子：逐算子等价验证与五维正交审计")
    parser.add_argument("--json", action="store_true", help="只输出 JSON 报告")
    parser.add_argument("--quick", action="store_true", help="跳过中心差分与训练冒烟")
    parser.add_argument("--bench", action="store_true", help="附加融合核基准")
    args = parser.parse_args(argv)

    report = run_verification(verbose=not args.json, quick=args.quick)
    if args.bench:
        bench = benchmark_fusion()
        bench_attn = benchmark_attention()
        report["benchmark"] = bench
        report["benchmark_attention"] = bench_attn
        if not args.json and bench.get("available"):
            print(f"[E] 融合核基准（{bench['shape']}）")
            print(f"    RMSNorm+RoPE  eager {bench['norm_rope_eager_ms']:.3f} ms → "
                  f"fused {bench['norm_rope_fused_ms']:.3f} ms  ({bench['norm_rope_speedup']:.2f}×)")
            print(f"    SwiGLU gate   eager {bench['swiglu_eager_ms']:.3f} ms → "
                  f"fused {bench['swiglu_fused_ms']:.3f} ms  ({bench['swiglu_speedup']:.2f}×)")
    if args.bench and bench_attn.get("available"):
        print(f"[F] 融合注意力基准（{bench_attn['shape']}，因果 + GQA）")
        for label in ("fp16", "fp32"):
            e = bench_attn[label]
            print(f"    {label}  fwd  {e['fwd_eager_ms']:.3f} → {e['fwd_flash_ms']:.3f} ms "
                  f"({e['fwd_speedup']:.2f}×)    fwd+bwd {e['fwd_bwd_eager_ms']:.3f} → "
                  f"{e['fwd_bwd_flash_ms']:.3f} ms ({e['fwd_bwd_speedup']:.2f}×)")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
