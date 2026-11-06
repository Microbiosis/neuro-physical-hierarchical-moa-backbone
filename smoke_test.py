"""
Clean-room smoke test for the Hybrid MoA Vision-Language Operator Suite.

Runs on CPU with numpy alone; the torch / Triton-simulator sections are skipped
automatically when torch is missing. No GPU and no Triton install required.

Checks, in order:

  1. the package imports and all 12 operator modules load under their
     canonical names
  2. numpy analytical kernels
       - branch RMSNorm forward/backward (+ finite-difference gradient check)
       - cross-modal anchor co-attention (bidirectional, zero distance decay)
       - cross-modal VL Hadamard-bilinear gated fusion (+ gradient check)
       - per-head Mixture-of-Attention mask (sink+window / dilated / full causal)
       - sparse top-k router
  3. torch nn.Module operators run forward + backward
       (CrossModalVisionLanguageFusionSubOp, HeterogeneousProposerOp,
        CrossProposalAggregatorOp in both modes, MixtureOfAttentionOp,
        UnifiedInterdisciplinaryMoALayer, two domain operators)
  4. Triton tile simulators reproduce the analytical references

Usage:
    python smoke_test.py            # run every check whose deps are available
    python smoke_test.py --quick    # imports + numpy kernels only

Exit code 0 = all available checks passed, 1 = at least one failed.
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np

PKG = "hybrid_moa_vl_operator_suite"
MODULES = [
    "moa_operators",
    "moa_triton_kernels",
    "unified_hybrid_moa_architecture",
    "quantum_tidal_operators",
    "neuro_nonlinear_operators",
    "local_self_modifying_operators",
    "supermembrane_spacetime_operators",
    "spinal_cord_pathway_operators",
    "tulpa_neuro_operators",
    "dissociative_trauma_neuro_operators",
    "corticospinal_chaotic_operators",
    "baryogenesis_parity_operators",
    "sleep_subconscious_neuro_operators",
    "vacuum_zpe_fluctuation_operators",
    "cpu_gpu_self_bootstrapping_operators",
    "quantum_resource_phasing_operators",
    "viral_nucleic_acid_invasion_operators",
]
# 独立算子模块（不含统一编排层，它按设计 import 全部兄弟模块）
STANDALONE = [m for m in MODULES if m != "unified_hybrid_moa_architecture"]

RESULTS: list[tuple[str, bool, str]] = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or "ok"))
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name}  {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def kernels():
    """NumpyMoAOperatorKernels from the package."""
    return importlib.import_module(f"{PKG}.moa_operators").NumpyMoAOperatorKernels


# --------------------------------------------------------------------------- #
# 1. imports
# --------------------------------------------------------------------------- #
def test_imports():
    pkg = importlib.import_module(PKG)
    assert pkg.__all__, "package exports nothing"
    bad = []
    for name in MODULES:
        try:
            importlib.import_module(f"{PKG}.{name}")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{name}: {exc}")
    if bad:
        raise RuntimeError("; ".join(bad))
    return f"{len(MODULES)}/{len(MODULES)} modules, {len(pkg.__all__)} exports"


def test_module_isolation():
    """Each standalone operator module imports in an empty directory."""
    import shutil

    pkg_dir = os.path.dirname(importlib.import_module(f"{PKG}.moa_operators").__file__)
    bad = []
    for name in STANDALONE:
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(os.path.join(pkg_dir, f"{name}.py"), td)
            r = subprocess.run(
                [sys.executable, "-c", f"import {name}"],
                cwd=td, capture_output=True, text=True, timeout=180,
            )
            if r.returncode != 0:
                tail = r.stderr.strip().splitlines()[-1] if r.stderr else "fail"
                bad.append(f"{name}: {tail}")
    if bad:
        raise RuntimeError("; ".join(bad))
    return f"{len(STANDALONE)}/{len(STANDALONE)} 模块可独立 import"


# --------------------------------------------------------------------------- #
# 2. numpy kernels
# --------------------------------------------------------------------------- #
def test_numpy_rmsnorm():
    k = kernels()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 8, 64))
    gamma = rng.standard_normal(64)

    out, inv_rms = k.branch_rmsnorm_forward(x, gamma)
    ref = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6) * gamma
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-12, f"forward mismatch {err}"

    dx, dgamma = k.branch_rmsnorm_backward(np.ones_like(out), x, gamma, inv_rms)
    assert dx.shape == x.shape and dgamma.shape == gamma.shape

    eps = 1e-6
    worst = 0.0
    for idx in [(0, 3, 5), (1, 7, 63), (0, 0, 0)]:
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num = (k.branch_rmsnorm_forward(xp, gamma)[0].sum()
               - k.branch_rmsnorm_forward(xm, gamma)[0].sum()) / (2 * eps)
        worst = max(worst, abs(num - float(dx[idx])) / max(1.0, abs(num)))
    assert worst < 1e-5, f"dx gradient off by {worst:.2e}"
    return f"fwd err={err:.1e}, grad rel-err={worst:.1e}"


def test_numpy_cross_modal_anchor_co_attention():
    """双向低秩跨模态锚点共注意力：零距离衰减 + 注意力归一化。"""
    k = kernels()
    rng = np.random.default_rng(1)
    B, L, D, R = 2, 12, 32, 4
    v = rng.standard_normal((B, L, D))
    t = rng.standard_normal((B, L, D))
    anchors = rng.standard_normal((R, D))

    c_align, cache = k.cross_modal_anchor_co_attention_forward(v, t, anchors)
    assert c_align.shape == (B, L, D), f"c_align {c_align.shape}"

    # 独立参考：pool 到 R 个锚点，再广播回每个 token
    scale = 1.0 / math.sqrt(D)
    joint = 0.5 * (v + t)
    sp = np.einsum("rd,bld->brl", anchors, joint) * scale
    ap = np.exp(sp - sp.max(-1, keepdims=True))
    ap = ap / ap.sum(-1, keepdims=True)                       # [B, R, L]
    assert np.allclose(ap.sum(axis=-1), 1.0, atol=1e-12), "pool attention must normalise over L"
    z = np.einsum("brl,bld->brd", ap, v)                      # [B, R, D]
    sb = np.einsum("bld,brd->blr", t, z) * scale
    ab = np.exp(sb - sb.max(-1, keepdims=True))
    ab = ab / ab.sum(-1, keepdims=True)                       # [B, L, R]
    assert np.allclose(ab.sum(axis=-1), 1.0, atol=1e-12), "broadcast attention must normalise over R"
    ref = np.einsum("blr,brd->bld", ab, z)
    err = float(np.max(np.abs(c_align - ref)))
    assert err < 1e-12, f"anchor co-attention mismatch {err}"

    # 零距离衰减：序列两端的 token 拿到的锚点上下文分布应无系统性衰减
    head = np.linalg.norm(c_align[:, :2], axis=-1).mean()
    tail = np.linalg.norm(c_align[:, -2:], axis=-1).mean()
    assert 0.5 < head / tail < 2.0, f"距离衰减异常 head/tail={head / tail:.3f}"
    return f"err={err:.1e}, 两端上下文比 head/tail={head / tail:.3f}"


def test_numpy_vl_bilinear_fusion():
    """Hadamard-Bilinear 跨模态门控合成：公式复算 + 边界 + 反向梯度。"""
    k = kernels()
    rng = np.random.default_rng(2)
    B, L, D, R = 2, 8, 32, 4
    v = rng.standard_normal((B, L, D))
    t = rng.standard_normal((B, L, D))
    anchors = rng.standard_normal((R, D))
    c_align, _ = k.cross_modal_anchor_co_attention_forward(v, t, anchors)
    av = rng.standard_normal(D) * 0.5
    at = rng.standard_normal(D) * 0.5
    bb = rng.standard_normal(D) * 0.5
    lam = 0.35

    out, cache = k.cross_modal_vl_bilinear_fusion_forward(v, t, c_align, av, at, bb, lam)
    assert out.shape == (B, L, D) and np.isfinite(out).all()

    # 公式复算
    g = 1.0 / (1.0 + np.exp(-np.clip(av * v + at * t, -500, 500)))
    y_interp = g * v + (1 - g) * t
    b_num = v * t + c_align
    inv_norm = 1.0 / np.sqrt(1.0 + lam * (v ** 2 + t ** 2))
    ref = y_interp + bb * b_num * inv_norm
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-12, f"bilinear fusion mismatch {err}"

    # 边界 1：beta=0 → 纯门控插值
    out_b0, _ = k.cross_modal_vl_bilinear_fusion_forward(v, t, c_align, av, at, np.zeros(D), lam)
    assert float(np.max(np.abs(out_b0 - y_interp))) < 1e-12, "beta=0 时应退化为门控插值"

    # 边界 2：alpha 拉到 ±500 → 门应几乎二值化（混合区 < 1%），且仍满足闭式解
    out_v, _ = k.cross_modal_vl_bilinear_fusion_forward(
        v, t, c_align, np.full(D, 500.0), np.full(D, -500.0), bb, lam)
    g_sat = 1.0 / (1.0 + np.exp(-np.clip(500.0 * v - 500.0 * t, -500, 500)))
    mixed = float(np.mean((g_sat > 0.01) & (g_sat < 0.99)))
    assert mixed < 0.01, f"饱和门下仍有 {mixed:.1%} 的混合区"
    expected = g_sat * v + (1 - g_sat) * t + bb * b_num * inv_norm
    err_sat = float(np.max(np.abs(out_v - expected)))
    assert err_sat < 1e-12, f"饱和门闭式解不符 {err_sat}"

    # 反向：梯度字典 + 对 v_vis 做有限差分抽查
    grads = k.cross_modal_vl_bilinear_fusion_backward(np.ones_like(out), cache)
    assert isinstance(grads, dict) and grads, "backward returned nothing"
    dv = None
    for key in ("d_vvis", "dv_vis", "grad_v_vis"):
        if key in grads:
            dv = grads[key]
            break
    if dv is None:
        raise AssertionError(f"backward 缺少 v_vis 梯度，键：{list(grads)}")
    assert dv.shape == v.shape
    eps = 1e-6
    i = (0, 3, 7)
    vp, vm = v.copy(), v.copy()
    vp[i] += eps
    vm[i] -= eps
    num = (k.cross_modal_vl_bilinear_fusion_forward(vp, t, c_align, av, at, bb, lam)[0].sum()
           - k.cross_modal_vl_bilinear_fusion_forward(vm, t, c_align, av, at, bb, lam)[0].sum()) / (2 * eps)
    rel = abs(num - float(dv[i])) / max(1.0, abs(num))
    assert rel < 1e-4, f"dv_vis 梯度偏差 {rel:.2e}"
    return f"fwd err={err:.1e}, 边界退化 ok, dv_vis 相对偏差={rel:.1e}"


def test_numpy_attention_mask():
    from importlib import import_module
    k = kernels()
    mask, stats = k.build_mixture_of_attention_mask(
        seq_len=32, n_heads=6, window_size=4, num_sinks=2, dilation=2)
    mask = np.asarray(mask)
    # v8 起返回 [n_heads, L, L]（不再带前导 batch 维）
    assert mask.shape == (6, 32, 32), f"mask shape {mask.shape}"
    assert mask.dtype == bool, f"mask dtype {mask.dtype}"
    tril = np.tril(np.ones((32, 32), dtype=bool))
    for h in range(6):
        assert np.all(mask[h] <= tril), f"head {h} violates causality"
    dens = mask.reshape(6, -1).mean(axis=1)
    full_causal = 32 * 33 / 2 / (32 * 32)
    assert dens.max() > full_causal - 1e-6, f"没有 full-causal 头（最大 {dens.max():.3f}）"
    assert dens.min() < 0.2, f"没有稀疏头（最小 {dens.min():.3f}）"
    assert len(set(np.round(dens, 6))) >= 2, "逐头掩码没有区分度"
    assert "overall_moa_density" in stats, f"stats keys {list(stats)}"
    return (f"per-head densities {[round(float(d), 3) for d in dens]} "
            f"(full-causal={full_causal:.3f}), overall={stats['overall_moa_density']:.3f}")


def test_numpy_router():
    k = kernels()
    rng = np.random.default_rng(3)
    B, N, L, D, E = 2, 4, 8, 32, 16
    x = rng.standard_normal((B, L, D))
    proposals = rng.standard_normal((B, N, L, D))
    w_rq = rng.standard_normal((D, E)) * 0.1
    w_rk = rng.standard_normal((D, E)) * 0.1

    out, aux, f_i = k.sparse_topk_router_forward(x, proposals, w_rq, w_rk, top_k=2)
    assert out.shape == (B, L, D) and np.isfinite(out).all()
    assert np.shape(f_i) == (N,), f"load stats {np.shape(f_i)}"
    assert math.isfinite(aux)
    return f"aux={aux:.4f}, 负载统计 shape={np.shape(f_i)}"


# --------------------------------------------------------------------------- #
# 3. torch operators
# --------------------------------------------------------------------------- #
def test_torch_vl_fusion():
    import torch

    mo = importlib.import_module(f"{PKG}.moa_operators")
    torch.manual_seed(0)
    op = mo.CrossModalVisionLanguageFusionSubOp(d_model=64, num_anchors=8)
    x = torch.randn(2, 12, 64, requires_grad=True)
    out = op(x)
    assert out.shape == x.shape, f"VL fusion out {tuple(out.shape)}"
    assert torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    gnorm = float(x.grad.norm())
    assert gnorm > 0.0, "no gradient reached the input"
    n = sum(p.numel() for p in op.parameters())
    return f"[B,L,D]={tuple(out.shape)}, {n / 1e3:.1f}K params, |grad|={gnorm:.4f}"


def test_torch_moa_stack():
    import torch

    mo = importlib.import_module(f"{PKG}.moa_operators")
    torch.manual_seed(1)
    cfg = mo.MoAOperatorConfig(d_model=64, n_heads=4, num_proposers=4,
                               top_k_proposers=2, d_ffn=128, num_cross_modal_anchors=8)
    x = torch.randn(2, 12, cfg.d_model)
    proposals = torch.randn(2, cfg.num_proposers, 12, cfg.d_model)

    prop = mo.HeterogeneousProposerOp(cfg)
    out = prop(x)
    assert out.shape == (2, cfg.num_proposers, 12, cfg.d_model), f"proposer out {tuple(out.shape)}"
    out.float().pow(2).mean().backward()

    attn_out, density = mo.MixtureOfAttentionOp(cfg)(x)
    assert attn_out.shape == x.shape and torch.isfinite(attn_out).all()
    assert 0.0 <= float(density) <= 1.0, f"density {density} out of range"

    for mode in ("cross_attention", "sparse_router"):
        cfg_m = mo.MoAOperatorConfig(d_model=64, n_heads=4, num_proposers=4,
                                     top_k_proposers=2, d_ffn=128,
                                     num_cross_modal_anchors=8, aggregator_mode=mode)
        synth, aux, metrics = mo.CrossProposalAggregatorOp(cfg_m)(x, proposals)
        assert synth.shape == x.shape, f"{mode} synth {tuple(synth.shape)}"
        assert torch.isfinite(synth).all()
        assert isinstance(metrics, dict) and metrics, f"{mode} returned no metrics"
    return f"proposer [B,N,L,D] ok, MoA density={float(density):.3f}, 2 aggregator modes ok"


def test_new_domain_ops():
    """v6/v7 新增的领域算子：睡眠/潜意识、真空零点能、Tri-ISA CPU-GPU 自举。"""
    import torch

    sl = importlib.import_module(f"{PKG}.sleep_subconscious_neuro_operators")
    vz = importlib.import_module(f"{PKG}.vacuum_zpe_fluctuation_operators")
    cg = importlib.import_module(f"{PKG}.cpu_gpu_self_bootstrapping_operators")
    qz = importlib.import_module(f"{PKG}.quantum_resource_phasing_operators")
    torch.manual_seed(6)
    x = torch.randn(2, 10, 64)
    specs = [
        ("sleep", sl.SleepSubconsciousConfig, sl.SleepSubconsciousNeuroOp,
         dict(d_model=64, n_heads=4, num_dream_anchors=8)),
        ("vacuum_zpe", vz.VacuumZPEFluctuationConfig, vz.VacuumZPEFluctuationOp,
         dict(d_model=64, n_heads=4, num_vacuum_modes=8)),
        ("cpu_gpu_bootstrap", cg.CPUGPUSelfBootstrappingConfig,
         cg.CPUGPUSelfBootstrappingOp, dict(d_model=64, n_heads=4, num_isa_anchors=8)),
        ("quantum_resource_phasing", qz.QuantumResourcePhasingConfig,
         qz.QuantumResourcePhasingOp, dict(d_model=64, n_heads=4)),
    ]
    for name, cfg_cls, op_cls, kw in specs:
        op = op_cls(cfg_cls(**kw))
        xin = x.clone().requires_grad_(True)
        out, diag = op(xin)
        assert out.shape == x.shape and torch.isfinite(out).all(), f"{name} 输出异常"
        out.float().pow(2).mean().backward()
        assert float(xin.grad.norm()) > 0.0, f"{name} 无梯度"
        assert isinstance(diag, dict) and diag, f"{name} 无诊断"
        n = sum(p.numel() for p in op.parameters())
        print(f"        · {name}: {n / 1e3:.1f}K params, diag keys={sorted(diag)[:4]}")
    return "sleep + vacuum_zpe + cpu_gpu_bootstrap + quantum_resource 前向/反向/诊断 ok"

def test_torch_unified_layer():
    import torch

    U = importlib.import_module(f"{PKG}.unified_hybrid_moa_architecture")
    if not getattr(U, "HAS_TORCH", False):
        return "skipped (torch layer disabled upstream)"
    torch.manual_seed(3)
    cfg = U.UnifiedHybridArchitectureConfig(d_model=64, n_heads=4, n_layers=2,
                                            num_proposers=3, top_k_proposers=2,
                                            num_vacuum_modes=8, num_cross_modal_anchors=8,
                                            num_local_groups=4, local_rank=2,
                                            num_extra_dims=3, num_tulpas=2,
                                            num_dream_anchors=8, num_isa_anchors=8)
    layer = U.UnifiedInterdisciplinaryMoALayer(cfg)
    x = torch.randn(2, 10, cfg.d_model, requires_grad=True)
    out, aux, diag = layer(x)
    assert out.shape == x.shape, f"unified out {tuple(out.shape)}"
    assert torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    gnorm = float(x.grad.norm())
    assert gnorm > 0.0, "no gradient through the unified layer"
    n = sum(p.numel() for p in layer.parameters())
    return f"15 算子 7 阶段前向 ok, {n / 1e3:.1f}K params, |grad|={gnorm:.4f}, diag keys={len(diag)}"


def test_torch_unified_backbone():
    """多层骨干：UnifiedHybridMoABackbone 堆叠 N 层 + final norm。"""
    import torch

    U = importlib.import_module(f"{PKG}.unified_hybrid_moa_architecture")
    if not getattr(U, "HAS_TORCH", False):
        return "skipped (torch layer disabled upstream)"
    torch.manual_seed(5)
    cfg = U.UnifiedHybridArchitectureConfig(d_model=64, n_heads=4, n_layers=3,
                                            num_proposers=3, top_k_proposers=2,
                                            num_vacuum_modes=8, num_cross_modal_anchors=8,
                                            num_local_groups=4, local_rank=2,
                                            num_extra_dims=3, num_tulpas=2,
                                            num_dream_anchors=8, num_isa_anchors=8)
    backbone = U.UnifiedHybridMoABackbone(cfg)
    x = torch.randn(2, 10, cfg.d_model, requires_grad=True)
    out, aux, telemetries = backbone(x)
    assert out.shape == x.shape, f"backbone out {tuple(out.shape)}"
    assert torch.isfinite(out).all()
    assert len(telemetries) == cfg.n_layers, f"telemetries {len(telemetries)} != n_layers {cfg.n_layers}"
    (out.float().pow(2).mean() + aux).backward()
    gnorm = float(x.grad.norm())
    assert gnorm > 0.0, "no gradient through the backbone"
    n = sum(p.numel() for p in backbone.parameters())
    return f"{cfg.n_layers} 层堆叠 ok, {n / 1e3:.1f}K params, |grad|={gnorm:.4f}"


def test_upstream_dissociative_suite():
    """跑包内自带的解离/创伤验证套件（7 变量 gradcheck + 极端冲击安全性）。"""
    import contextlib
    import io

    dt = importlib.import_module(f"{PKG}.dissociative_trauma_neuro_operators")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dt.run_dissociative_trauma_verification_suite()
    out = buf.getvalue()
    n_pass = out.count("[PASS")
    assert "FAIL" not in out, "上游验证套件出现 FAIL"
    assert n_pass >= 7, f"只看到 {n_pass} 个 PASS（预期 ≥7）"
    assert "Bounded" in out, "缺少极端冲击有界性检查"
    return f"上游套件 {n_pass} 项 PASS，冲击输出有界"


def test_mechanism_stage_decoupling():
    """机制审计（v6）：Stage 0/III/IV/V 与干净感觉运动流对创伤参数严格零灵敏 + 互补门控。"""
    U = importlib.import_module(f"{PKG}.unified_hybrid_moa_architecture")
    rng = np.random.default_rng(11)
    cfg = U.UnifiedHybridArchitectureConfig(d_model=64, n_heads=4, num_proposers=3,
                                            top_k_proposers=2, num_vacuum_modes=8,
                                            num_cross_modal_anchors=8, num_local_groups=4,
                                            local_rank=2, num_extra_dims=3, num_tulpas=2,
                                            num_dream_anchors=8, num_isa_anchors=8)
    B, L, D = 2, 16, cfg.d_model
    x = rng.standard_normal((B, L, D)) * 0.5
    params = U.UnifiedArchitectureExecutionEngine.init_layer_parameters(D, cfg, seed=2026)
    fwd = U.UnifiedArchitectureExecutionEngine.forward_unified_layer_exact

    _, ctx = fwd(x, params, cfg, use_smooth_router=True)

    # 1) 因果解耦：大幅扰动创伤参数，Stage 0/III/IV/V 与干净流必须逐位不变
    pert = {k: v.copy() for k, v in params.items()}
    pert["alpha_hyper"] += 10.0
    pert["gamma_dpdr"] += 10.0
    pert["theta_dissoc"] += 5.0
    _, ctx_p = fwd(x, pert, cfg, use_smooth_router=True)

    groups = {
        "Stage0.A VacuumZPE": ("zpe_out",),
        "Stage0.B Baryogenesis": ("baryon_out",),
        "StageI.D Molecular state": ("x_molecular",),
        "StageIII Cortex": ("h_cortex", "proposals_raw"),
        "StageIV CPUGPU_TriISA": ("boot_out",),
        "StageV Tulpa": ("tulpa_out",),
        "StageVI Sleep": ("y_consolidated",),
        "干净感觉运动流": ("reflex_bypass", "mega_syn"),
    }
    worst = 0.0
    for name, keys in groups.items():
        d = max(float(np.max(np.abs(ctx[k] - ctx_p[k]))) for k in keys)
        worst = max(worst, d)
        assert d == 0.0, f"{name} 对创伤参数有灵敏度 {d:.3e}"
    # 对照：创伤通路本身必须确实变了（证明扰动生效，不是空操作）
    d_trauma = float(np.max(np.abs(ctx["trauma_out"] - ctx_p["trauma_out"])))
    assert d_trauma > 1.0, f"扰动未生效，创伤输出只变了 {d_trauma:.3e}"

    # 2) 互补门控：常态 -> 意志/Tulpa/睡眠门占优；+50 冲击 -> 创伤防御门接管
    g_norm = float(np.mean(ctx["g_shut_gate"]))
    x_shock = x.copy()
    x_shock[:, 0, :] += 50.0
    _, ctx_s = fwd(x_shock, params, cfg, use_smooth_router=True)
    g_shock = float(np.mean(ctx_s["g_shut_gate"][:, 0, :]))
    assert g_norm > 0.70, f"常态门控 {g_norm:.4f} 未占优"
    assert g_shock < 1e-10, f"冲击下意志门未关闭 {g_shock:.3e}"

    # 3) 冲击触发 DPDR 感觉压缩（创伤表征被压小）
    tr_norm = float(np.linalg.norm(ctx["trauma_out"][:, 0, :]))
    tr_shock = float(np.linalg.norm(ctx_s["trauma_out"][:, 0, :]))
    assert tr_shock < tr_norm, f"冲击未触发 DPDR 压缩 {tr_shock:.3f} vs {tr_norm:.3f}"
    return (f"分子态与各干净通路对创伤参数的最大变化为 {worst:.1e}（创伤通路自身变了 {d_trauma:.1f}），"
            f"G_shut {g_norm:.4f}→{g_shock:.2e}，DPDR 压缩 {tr_norm / max(tr_shock, 1e-12):.1f}×")


def test_domain_ops():
    import torch

    qt = importlib.import_module(f"{PKG}.quantum_tidal_operators")
    nn_mod = importlib.import_module(f"{PKG}.neuro_nonlinear_operators")
    torch.manual_seed(4)
    x = torch.randn(2, 10, 64)
    for name, op in [
        ("quantum_tidal", qt.QuantumTidalFluctuationOp(qt.QuantumTidalConfig(d_model=64, n_heads=4))),
        ("neuro", nn_mod.NonlinearDendriticNeuroOp(nn_mod.NeuroNonlinearConfig(d_model=64, n_heads=4))),
    ]:
        out, diag = op(x)
        assert out.shape == x.shape, f"{name} out {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name} output not finite"
        assert isinstance(diag, dict) and diag, f"{name} returned no diagnostics"
    return "quantum_tidal + neuro forward ok, both return (out, diagnostics)"


# --------------------------------------------------------------------------- #
# 4. triton tile simulators
# --------------------------------------------------------------------------- #
def test_triton_simulators():
    tk = importlib.import_module(f"{PKG}.moa_triton_kernels")
    k = kernels()
    sim = tk.TritonTileSimulator()
    rng = np.random.default_rng(7)
    B, L, D, R, N = 2, 16, 32, 4, 4

    v = rng.standard_normal((B, L, D))
    t = rng.standard_normal((B, L, D))
    c, _ = k.cross_modal_anchor_co_attention_forward(v, t, rng.standard_normal((R, D)))
    av = rng.standard_normal(D) * 0.5
    at = rng.standard_normal(D) * 0.5
    bb = rng.standard_normal(D) * 0.5
    y_sim, rep_b = sim.simulate_fused_cross_modal_vl_bilinear(v, t, c, av, at, bb)
    y_ref, _ = k.cross_modal_vl_bilinear_fusion_forward(v, t, c, av, at, bb)
    err_b = float(np.max(np.abs(y_sim - y_ref)))
    assert err_b < 1e-10, f"VL bilinear simulator mismatch {err_b}"

    proposals = rng.standard_normal((B, N, L, D))
    gamma = rng.standard_normal(D)
    scale = rng.standard_normal((N, D))
    logits = rng.standard_normal((B, L, N))
    out_sim, rep_m = sim.simulate_persistent_norm_sparse_topk_megakernel(
        proposals, gamma, scale, logits, top_k=2)
    assert out_sim.shape == (B, L, D) and np.isfinite(out_sim).all()

    idx = np.argpartition(logits, -2, axis=-1)[..., -2:]
    vals = np.take_along_axis(logits, idx, axis=-1)
    w = np.exp(vals - vals.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    normed = proposals / np.sqrt(np.mean(proposals ** 2, -1, keepdims=True) + 1e-6)
    normed = normed * gamma[None, None, None, :] * scale[None, :, None, :]
    ref = np.zeros((B, L, D))
    for kk in range(2):
        sel = idx[:, :, kk]
        ref += w[:, :, kk:kk + 1] * np.take_along_axis(
            np.transpose(normed, (0, 2, 1, 3)), sel[:, :, None, None], axis=2)[:, :, 0]
    err_m = float(np.max(np.abs(out_sim - ref)))
    assert err_m < 1e-10, f"megakernel simulator mismatch {err_m}"

    for tag, rep in [("vl_bilinear", rep_b), ("megakernel", rep_m)]:
        d = getattr(rep, "max_abs_diff_vs_ref", None)
        assert d is not None and d < 1e-10, f"{tag} self-reported diff {d}"
    bw_b = getattr(rep_b, "bandwidth_reduction_pct", None)
    return (f"bilinear err={err_b:.1e}, megakernel err={err_m:.1e}, "
            f"自报带宽节省={bw_b:.1f}%" if bw_b is not None else f"errs {err_b:.1e}/{err_m:.1e}")


def test_numpy_viral_operator():
    from hybrid_moa_vl_operator_suite import ViralInvasionHybridMoABackbone, ViralInvasionOperatorConfig

    rng = np.random.default_rng(162)
    x = 0.5 * rng.normal(size=(2, 4, 8))
    dx_probe = rng.normal(size=x.shape)
    dy = rng.normal(size=x.shape)
    eps = 1e-6
    for adaptive in (True, False):
        op = ViralInvasionHybridMoABackbone(ViralInvasionOperatorConfig(
            d_model=8, num_refinement_steps=3, share_refinement_weights=adaptive,
            enable_adaptive_halting=adaptive, halting_threshold=0.55,
        ))
        params = op.named_parameters()
        _, cache = op.forward(x)
        op.forward(x * 0.8)  # A second forward must leave the first context usable.
        dx, grads = op.backward(dy, cache)
        directions = {key: rng.normal(size=value.shape) for key, value in params.items()}
        up, _ = op.forward(x + eps * dx_probe, {k: v + eps * directions[k] for k, v in params.items()})
        down, _ = op.forward(x - eps * dx_probe, {k: v - eps * directions[k] for k, v in params.items()})
        analytical = np.sum(dx * dx_probe) + sum(np.sum(grads[k] * d) for k, d in directions.items())
        numerical = np.sum(dy * (up - down)) / (2.0 * eps)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-8)
        if adaptive:
            active = cache["halting_probs"][0] < op.config.halting_threshold
            assert active.any() and not active.all()
    return "input and parameter gradients match finite differences in both halting modes"


def test_numpy_required_roles():
    from hybrid_moa_vl_operator_suite import (
        UnifiedArchitectureExecutionEngine as Engine,
        UnifiedHybridArchitectureConfig,
    )

    cfg = UnifiedHybridArchitectureConfig(
        d_model=16, n_heads=2, num_local_groups=2, local_rank=2, top_k_proposers=1,
    )
    rng = np.random.default_rng(26)
    x = 0.5 * rng.normal(size=(3, 9, 16))
    dy = rng.normal(size=x.shape)
    params = Engine.init_layer_parameters(cfg.d_model, cfg)
    forward = Engine.forward_unified_layer_exact
    y, ctx = forward(x, params, cfg)
    assert tuple(ctx["role_outputs"]) == cfg.STAGE_DOMAIN_NAMES
    viral_role = next(name for name in cfg.STAGE_DOMAIN_NAMES if name.startswith("StageI.D:"))
    np.testing.assert_array_equal(ctx["x_molecular"], ctx["x_spacetime"] * (0.5 + ctx["molecular_prob"]))
    for role_name in cfg.STAGE_DOMAIN_NAMES:
        ablated, ablated_ctx = forward(x, params, cfg, ablate_stage_name=role_name)
        assert np.isfinite(ablated).all() and not np.allclose(ablated, y), role_name
        if role_name == viral_role:
            np.testing.assert_array_equal(ablated_ctx["x_molecular"], ctx["x_spacetime"])
            np.testing.assert_array_equal(ablated_ctx["trauma_out"], ctx["trauma_out"])
            assert not np.allclose(ablated_ctx["spinal_out"], ctx["spinal_out"])

    # Each cortical role must affect the output even when every routed value is zero.
    no_route = {**params, "scale_norm": np.zeros_like(params["scale_norm"])}
    y_roles, roles_ctx = forward(x, no_route, cfg)
    cortical_roles = [name for name in cfg.STAGE_DOMAIN_NAMES if name.startswith("StageIII.Col")]
    for role_name in cortical_roles:
        ablated, _ = forward(x, no_route, cfg, ablate_stage_name=role_name)
        assert not np.allclose(ablated, y_roles), role_name
    alignment, excitability, _ = cortical_roles
    roles = roles_ctx["role_outputs"]
    swapped, _ = forward(x, no_route, cfg, role_replacements={
        alignment: roles[excitability], excitability: roles[alignment],
    })
    assert not np.allclose(swapped, y_roles)

    saturated = {**params, "stage1c_qrp_gate": np.array(-1000.0)}
    _, gate_ctx = forward(x, saturated, cfg)
    assert gate_ctx["mixing_gates"]["stage1c_qrp_gate"] >= 0.05
    grads = Engine.backward_unified_layer_exact(dy, ctx)
    eps = 1e-6
    direction = rng.normal(size=x.shape)
    up, _ = forward(x + eps * direction, params, cfg)
    down, _ = forward(x - eps * direction, params, cfg)
    np.testing.assert_allclose(
        np.sum(grads["d_x_in"] * direction), np.sum(dy * (up - down)) / (2.0 * eps), rtol=1e-6, atol=1e-8,
    )
    key = "stage1d_viral_router_weight"
    direction = rng.normal(size=params[key].shape)
    up, _ = forward(x, {**params, key: params[key] + eps * direction}, cfg)
    down, _ = forward(x, {**params, key: params[key] - eps * direction}, cfg)
    np.testing.assert_allclose(
        np.sum(grads[f"d_{key}"] * direction), np.sum(dy * (up - down)) / (2.0 * eps), rtol=1e-6, atol=1e-8,
    )
    return "15 ablations, cortical role swap with zero routed values, Stage I.D and input gradchecks"


def test_torch_viral_operator():
    import torch
    from hybrid_moa_vl_operator_suite import TorchViralNucleicAcidInvasionOperator, ViralInvasionOperatorConfig

    torch.manual_seed(42)
    op = TorchViralNucleicAcidInvasionOperator(ViralInvasionOperatorConfig(d_model=8)).double()
    x1 = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    x2 = torch.randn_like(x1, requires_grad=True)
    y1, y2 = op(x1), op(x2)
    params = {name: param.detach().numpy() for name, param in op.weights.items()}
    ref1, cache1 = op.core.forward(x1.detach().numpy(), params)
    ref2, cache2 = op.core.forward(x2.detach().numpy(), params)
    np.testing.assert_array_equal(y1.detach().numpy(), ref1)
    np.testing.assert_array_equal(y2.detach().numpy(), ref2)
    dx1, grads1 = op.core.backward(2.0 * ref1, cache1)
    dx2, grads2 = op.core.backward(2.0 * ref2, cache2)
    (y1.square().sum() + y2.square().sum()).backward()
    np.testing.assert_allclose(x1.grad.numpy(), dx1, atol=1e-14)
    np.testing.assert_allclose(x2.grad.numpy(), dx2, atol=1e-14)
    for name, param in op.weights.items():
        np.testing.assert_allclose(param.grad.numpy(), grads1[name] + grads2[name], atol=1e-14)
    return "two forwards on one module, both outputs and all gradients match NumPy"


def test_torch_required_role_backbone():
    import torch
    from hybrid_moa_vl_operator_suite import (
        UnifiedHybridArchitectureConfig, UnifiedHybridMoABackbone,
    )

    torch.manual_seed(43)
    cfg = UnifiedHybridArchitectureConfig(
        d_model=16, n_heads=2, n_layers=2, num_local_groups=2, local_rank=2, top_k_proposers=1,
    )
    model = UnifiedHybridMoABackbone(cfg)
    model.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        for layer in model.layers:
            layer.branch_scale_4d.zero_()

    x = torch.randn(2, 4, 16, requires_grad=True)
    y, aux, telemetry = model(x)
    assert y.shape == x.shape and y.isfinite().all()
    assert len(telemetry) == len(model.layers) and all(t["stage1d_molecular_rms"] > 0 for t in telemetry)
    (y[..., 0].square().mean() + aux).backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert all(param.grad is None or param.grad.isfinite().all() for param in model.parameters())
    for layer in model.layers:
        required_operators = (
            layer.stage0a_vacuum_zpe, layer.stage0b_baryogenesis, layer.stage1a_supermembrane,
            layer.stage1b_quantum_tidal, layer.stage1c_quantum_resource_phasing, layer.stage1d_viral_invasion,
            layer.stage2a_spinal_pathway, layer.stage2b_corticospinal_chaos, layer.stage2c_trauma_regulator,
            layer.col0_vl_association, layer.col1_dendritic_nmda, layer.col2_synaptic_selfmod,
            layer.stage4_cpu_gpu_bootstrap, layer.stage5_tulpa_agency, layer.stage6_sleep_subconscious,
        )
        parameter_ids = [id(param) for op in required_operators for param in op.parameters()]
        assert len(parameter_ids) == len(set(parameter_ids))
        for op in required_operators:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in op.parameters()), type(op).__name__
        weights = layer.stage1d_viral_invasion.weights
        for name in ("router_weight", "stage_gates_raw", "refinement_gates_raw", "halting_probe_weight", "stage1_alpha_trop"):
            assert weights[name].grad.abs().sum() > 0, name
    return "all 15 roles train in both layers with routed values zero"


def main() -> int:
    quick = "--quick" in sys.argv
    torch_ok = has_torch()
    print("Hybrid MoA Vision-Language Operator Suite — smoke test")
    print(f"package: {PKG} | numpy {np.__version__}"
          + (", torch available" if torch_ok else ", torch MISSING (torch checks skipped)"))
    print("-" * 74)

    check("package imports and exports", test_imports)
    check("standalone module imports", test_module_isolation)
    check("numpy branch RMSNorm fwd/bwd + FD gradient", test_numpy_rmsnorm)
    check("numpy cross-modal anchor co-attention (zero decay)", test_numpy_cross_modal_anchor_co_attention)
    check("numpy VL Hadamard-bilinear fusion + gradient", test_numpy_vl_bilinear_fusion)
    check("numpy per-head MoA attention mask", test_numpy_attention_mask)
    check("numpy sparse top-k router", test_numpy_router)
    check("numpy viral invasion operator forward/backward", test_numpy_viral_operator)
    check("numpy unified layer required roles", test_numpy_required_roles)

    if not quick and torch_ok:
        check("torch CrossModalVisionLanguageFusionSubOp fwd/bwd", test_torch_vl_fusion)
        check("torch MoA stack (proposer + attention + 2 aggregators)", test_torch_moa_stack)
        check("torch UnifiedInterdisciplinaryMoALayer (15 operators / 7 stages)", test_torch_unified_layer)
        check("torch UnifiedHybridMoABackbone (3 layers)", test_torch_unified_backbone)
        check("torch viral invasion operator parity and cache isolation", test_torch_viral_operator)
        check("torch UnifiedHybridMoABackbone required roles", test_torch_required_role_backbone)
        check("torch new domain operators (sleep, vacuum ZPE, CPU-GPU, quantum resource)", test_new_domain_ops)
        check("torch domain operators (quantum tidal, neuro)", test_domain_ops)
        check("upstream dissociative verification suite", test_upstream_dissociative_suite)
        check("mechanism: Stage 0/III/IV/V/VI vs Trauma decoupling + regime gates",
              test_mechanism_stage_decoupling)
        check("triton tile simulators vs references", test_triton_simulators)
    elif not quick:
        print("  SKIP  torch / simulator sections (torch not installed)")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("-" * 74)
    print(f"{passed}/{total} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
