"""
Clean-room smoke test for the Neuro-Physical Hierarchical MoA Backbone library.

Everything here runs on CPU with numpy alone, except the torch sections which
are skipped automatically when torch is missing. No GPU, no Triton required.

Checks, in order:

  1. all 12 modules import under their canonical snake_case names
  2. numpy analytical kernels
       - branch RMSNorm forward/backward (+ finite-difference gradient check)
       - 4 SSM scan implementations agree with the sequential reference
       - per-head Mixture-of-Attention mask (sink+window / dilated / full causal)
       - sparse top-k router and cross-proposal attention
  3. torch nn.Module operators run forward + backward
       (HeterogeneousProposerOp / MixtureOfAttentionOp / CrossProposalAggregatorOp
        and two domain operators)
  4. Triton tile simulators reproduce their analytical references bit-accurately

Usage:
    python smoke_test.py            # run every check whose deps are available
    python smoke_test.py --quick    # imports + numpy kernels only

Exit code 0 = all available checks passed, 1 = at least one failed.
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np

MODULES = [
    "baryogenesis_parity_operators",
    "corticospinal_chaotic_operators",
    "dissociative_trauma_neuro_operators",
    "local_self_modifying_operators",
    "moa_operators",
    "moa_triton_kernels",
    "neuro_nonlinear_operators",
    "quantum_tidal_operators",
    "spinal_cord_pathway_operators",
    "supermembrane_spacetime_operators",
    "tulpa_neuro_operators",
    "unified_hybrid_moa_architecture",
]

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


# --------------------------------------------------------------------------- #
# 1. imports
# --------------------------------------------------------------------------- #
def test_imports():
    import importlib

    bad = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{name}: {exc}")
    if bad:
        raise RuntimeError("; ".join(bad))
    return f"{len(MODULES)}/{len(MODULES)} modules"


# --------------------------------------------------------------------------- #
# 2. numpy kernels
# --------------------------------------------------------------------------- #
def _ssm_inputs(B=2, L=16, D=8, N=4, seed=0):
    """Discrete selective-SSM tensors shared by all four scan implementations."""
    rng = np.random.default_rng(seed)
    dt = rng.uniform(0.05, 0.5, size=(B, L, D))
    u = rng.standard_normal((B, L, D))
    A = -np.exp(rng.uniform(-2.0, 0.0, size=(D, N)))
    B_ssm = rng.standard_normal((B, L, N)) * 0.2
    C_ssm = rng.standard_normal((B, L, N)) * 0.2
    dA = np.exp(dt[:, :, :, None] * A[None, None, :, :])       # [B, L, D, N]
    dB_u = dt[:, :, :, None] * u[:, :, :, None] * B_ssm[:, :, None, :]
    return dict(u=u, dt=dt, A=A, B_ssm=B_ssm, C_ssm=C_ssm, dA=dA, dB_u=dB_u)


def test_numpy_rmsnorm():
    from moa_operators import NumpyMoAOperatorKernels

    k = NumpyMoAOperatorKernels()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 8, 64))
    gamma = rng.standard_normal(64)

    out, inv_rms = k.branch_rmsnorm_forward(x, gamma)
    ref = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6) * gamma
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-12, f"forward mismatch {err}"
    assert np.allclose(inv_rms, 1.0 / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + 1e-6))

    dx, dgamma = k.branch_rmsnorm_backward(np.ones_like(out), x, gamma, inv_rms)
    assert dx.shape == x.shape and dgamma.shape == gamma.shape

    # finite-difference check of dx on a handful of elements
    eps = 1e-6
    worst = 0.0
    for idx in [(0, 3, 5), (1, 7, 63), (0, 0, 0), (1, 4, 31)]:
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num = (k.branch_rmsnorm_forward(xp, gamma)[0].sum()
               - k.branch_rmsnorm_forward(xm, gamma)[0].sum()) / (2 * eps)
        worst = max(worst, abs(num - float(dx[idx])) / max(1.0, abs(num)))
    assert worst < 1e-5, f"dx gradient off by {worst:.2e}"
    return f"fwd err={err:.1e}, grad rel-err={worst:.1e}"


def test_numpy_ssm_scans():
    from moa_operators import NumpyMoAOperatorKernels

    k = NumpyMoAOperatorKernels()
    s = _ssm_inputs()
    y_seq, _ = k.ssm_sequential_scan_ref(s["dA"], s["dB_u"], s["C_ssm"])
    y_ks, _, _ = k.ssm_kogge_stone_parallel_scan(s["dA"], s["dB_u"], s["C_ssm"])
    y_cs, _ = k.ssm_chunkwise_ssd_scan(np.log(s["dA"]), s["dB_u"], s["C_ssm"], chunk_size=4)
    y_zo, h_bnd, stats = k.ssm_zero_oom_checkpointed_forward(
        s["u"], s["dt"], s["A"], s["B_ssm"], s["C_ssm"], chunk_size=4, state_tile=4
    )
    worst = 0.0
    for name, y in [("kogge_stone", y_ks), ("chunkwise_ssd", y_cs), ("zero_oom", y_zo)]:
        assert y.shape == y_seq.shape, f"{name} shape {y.shape} != {y_seq.shape}"
        worst = max(worst, float(np.max(np.abs(y - y_seq))))
    assert worst < 1e-10, f"scan disagreement {worst}"
    assert h_bnd.shape[1] == 4, f"expected 4 chunk boundaries, got {h_bnd.shape}"
    return f"4 scans agree, max|diff|={worst:.1e}, boundaries={h_bnd.shape}"


def test_numpy_attention_mask():
    from moa_operators import NumpyMoAOperatorKernels

    k = NumpyMoAOperatorKernels()
    mask, stats = k.build_mixture_of_attention_mask(
        seq_len=32, n_heads=6, window_size=4, num_sinks=2, dilation=2
    )
    mask = np.asarray(mask)
    # [1, n_heads, L, L] — leading singleton batch dim broadcasts over the batch
    assert mask.shape == (1, 6, 32, 32), f"mask shape {mask.shape}"
    assert mask.dtype == bool, f"mask dtype {mask.dtype}"
    tril = np.tril(np.ones((32, 32), dtype=bool))
    for h in range(6):
        assert np.all(mask[0, h] <= tril), f"head {h} violates causality"
    # head modes cycle sink+window / dilated / full causal -> densities must differ
    dens = mask[0].reshape(6, -1).mean(axis=1)
    assert dens[2] > dens[0], f"full-causal head {dens[2]:.3f} not denser than windowed {dens[0]:.3f}"
    assert dens[0] < dens[1] or dens[1] < dens[2], "per-head masks should not all be identical"
    assert "overall_moa_density" in stats, f"stats keys {list(stats)}"
    return f"per-head densities {[round(float(d), 3) for d in dens]}, overall={stats['overall_moa_density']:.3f}"


def test_numpy_router_and_attention():
    from moa_operators import NumpyMoAOperatorKernels

    k = NumpyMoAOperatorKernels()
    rng = np.random.default_rng(3)
    B, N, L, D, E = 2, 4, 8, 32, 16
    x = rng.standard_normal((B, L, D))
    proposals = rng.standard_normal((B, N, L, D))
    w_rq = rng.standard_normal((D, E)) * 0.1
    w_rk = rng.standard_normal((D, E)) * 0.1

    out, aux, f_i = k.sparse_topk_router_forward(x, proposals, w_rq, w_rk, top_k=2)
    assert out.shape == (B, L, D), f"router out {out.shape}"
    assert np.isfinite(out).all()
    assert f_i.shape == (N,), f"load stats {np.shape(f_i)}"
    assert math.isfinite(aux)

    w_q = rng.standard_normal((D, D)) * 0.1
    w_k = rng.standard_normal((D, D)) * 0.1
    w_v = rng.standard_normal((D, D)) * 0.1
    w_o = rng.standard_normal((D, D)) * 0.1
    agg, attn = k.cross_proposal_attention_forward(x, proposals, w_q, w_k, w_v, w_o, n_heads=4)
    assert agg.shape == (B, L, D)
    attn = np.asarray(attn)
    # [B, L, n_heads, 1, N] — one query slot attending over the N proposers
    assert attn.shape == (B, L, 4, 1, N), f"attn shape {attn.shape}"
    assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-10), "attention must sum to 1 over proposers"

    # fused per-proposer GEMM: x [B, L, D] x weight [N, D, O] -> [B, N, L, O]
    w_p = rng.standard_normal((N, D, D)) * 0.1
    b_p = rng.standard_normal((N, D)) * 0.1
    y = k.fused_proposer_linear_forward(x, w_p, b_p)
    assert y.shape == (B, N, L, D), f"proposer linear out {y.shape}"
    ref = np.einsum("bld,ndo->bnlo", x, w_p) + b_p[None, :, None, :]
    assert float(np.max(np.abs(y - ref))) < 1e-12

    gx, gw, gb = k.fused_proposer_linear_backward(np.ones_like(y), x, w_p)
    assert gx.shape == x.shape and gw.shape == w_p.shape and gb.shape == b_p.shape
    # gradient check on the bias of proposer 0
    eps = 1e-6
    i = (0, 0)
    bp, bm = b_p.copy(), b_p.copy()
    bp[i] += eps
    bm[i] -= eps
    num = (k.fused_proposer_linear_forward(x, w_p, bp).sum()
           - k.fused_proposer_linear_forward(x, w_p, bm).sum()) / (2 * eps)
    assert abs(num - float(gb[i])) < 1e-4 * max(1.0, abs(num)), f"gb {num} vs {gb[i]}"
    return f"router aux={aux:.4f}, attn sums to 1 over {N} proposers, fused GEMM+grad ok"


# --------------------------------------------------------------------------- #
# 3. torch operators
# --------------------------------------------------------------------------- #
def test_torch_proposer():
    import torch

    from moa_operators import HeterogeneousProposerOp, MoAOperatorConfig

    torch.manual_seed(0)
    cfg = MoAOperatorConfig(d_model=64, n_heads=4, num_proposers=4, top_k_proposers=2,
                            d_ffn=128, d_state=8)
    x = torch.randn(2, 12, cfg.d_model, requires_grad=True)

    op = HeterogeneousProposerOp(cfg)
    out = op(x)
    assert out.shape == (2, cfg.num_proposers, 12, cfg.d_model), \
        f"proposer out {tuple(out.shape)} (expected [B, N, L, D])"
    out.float().pow(2).mean().backward()
    gnorm = float(x.grad.norm())
    assert gnorm > 0.0, "no gradient reached the input"
    n_params = sum(p.numel() for p in op.parameters())
    return f"[B,N,L,D]={tuple(out.shape)}, {n_params / 1e3:.1f}K params, |grad|={gnorm:.3f}"


def test_torch_moa_stack():
    import torch

    from moa_operators import CrossProposalAggregatorOp, MixtureOfAttentionOp, MoAOperatorConfig

    torch.manual_seed(1)
    cfg = MoAOperatorConfig(d_model=64, n_heads=4, num_proposers=4, top_k_proposers=2,
                            d_ffn=128, d_state=8)
    x = torch.randn(2, 12, cfg.d_model)
    proposals = torch.randn(2, cfg.num_proposers, 12, cfg.d_model)

    attn_out, density = MixtureOfAttentionOp(cfg)(x)
    assert attn_out.shape == x.shape and torch.isfinite(attn_out).all()
    assert 0.0 <= float(density) <= 1.0, f"density {density} out of range"

    for mode in ("cross_attention", "sparse_router"):
        cfg_m = MoAOperatorConfig(d_model=64, n_heads=4, num_proposers=4,
                                  top_k_proposers=2, d_ffn=128, d_state=8,
                                  aggregator_mode=mode)
        synth, aux, metrics = CrossProposalAggregatorOp(cfg_m)(x, proposals)
        assert synth.shape == x.shape, f"{mode} synth {tuple(synth.shape)}"
        assert torch.isfinite(synth).all()
        assert isinstance(metrics, dict) and metrics, f"{mode} returned no metrics"
    return f"MoA attn density={float(density):.3f}, 2 aggregator modes ok"


def test_domain_ops():
    import torch

    from neuro_nonlinear_operators import NeuroNonlinearConfig, NonlinearDendriticNeuroOp
    from quantum_tidal_operators import QuantumTidalConfig, QuantumTidalFluctuationOp

    torch.manual_seed(2)
    x = torch.randn(2, 10, 64)
    for name, op in [
        ("neuro", NonlinearDendriticNeuroOp(NeuroNonlinearConfig(d_model=64, n_heads=4))),
        ("tidal", QuantumTidalFluctuationOp(QuantumTidalConfig(d_model=64, n_heads=4))),
    ]:
        out, diag = op(x)
        assert out.shape == x.shape, f"{name} out {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name} output not finite"
        assert isinstance(diag, dict) and diag, f"{name} returned no diagnostics"
    return "neuro + tidal forward ok, both return (out, diagnostics)"


# --------------------------------------------------------------------------- #
# 4. triton tile simulators
# --------------------------------------------------------------------------- #
def test_triton_simulators():
    from moa_operators import NumpyMoAOperatorKernels
    from moa_triton_kernels import TritonTileSimulator

    sim = TritonTileSimulator()
    rng = np.random.default_rng(7)
    B, N, L, D, Ns = 2, 4, 16, 32, 4

    proposals = rng.standard_normal((B, N, L, D))
    gamma = rng.standard_normal(D)
    scale = rng.standard_normal((N, D))
    router_logits = rng.standard_normal((B, L, N))
    out, report = sim.simulate_persistent_norm_sparse_topk_megakernel(
        proposals, gamma, scale, router_logits, top_k=2
    )
    assert out.shape == (B, L, D) and np.isfinite(out).all()

    # independent reference: normalise each branch, then weighted top-k reduce
    idx = np.argpartition(router_logits, -2, axis=-1)[..., -2:]
    vals = np.take_along_axis(router_logits, idx, axis=-1)
    w = np.exp(vals - vals.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    normed = proposals / np.sqrt(np.mean(proposals ** 2, -1, keepdims=True) + 1e-6)
    normed = normed * gamma[None, None, None, :] * scale[None, :, None, :]
    ref = np.zeros((B, L, D))
    for k in range(2):
        sel = idx[:, :, k]
        ref += w[:, :, k:k + 1] * np.take_along_axis(
            np.transpose(normed, (0, 2, 1, 3)), sel[:, :, None, None], axis=2
        )[:, :, 0]
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-10, f"megakernel mismatch {err}"

    u = rng.standard_normal((B, L, D))
    z = rng.standard_normal((B, L, D))
    dt_raw = rng.standard_normal((B, L, D))
    B_ssm = rng.standard_normal((B, L, Ns)) * 0.2
    C_ssm = rng.standard_normal((B, L, Ns)) * 0.2
    A_log = rng.uniform(-2.0, 0.0, size=(D, Ns))
    dt_bias = rng.standard_normal(D) * 0.1
    D_skip = rng.standard_normal(D) * 0.1
    y, rep2 = sim.simulate_fused_selective_ssm_parallel_scan(
        u, z, dt_raw, B_ssm, C_ssm, A_log, dt_bias, D_skip
    )
    assert y.shape == (B, L, D) and np.isfinite(y).all()

    # the report is the library's own agreement measure against its reference
    for tag, rep in [("megakernel", report), ("ssm_scan", rep2)]:
        d = getattr(rep, "max_abs_diff_vs_ref", None)
        assert d is not None and d < 1e-10, f"{tag} self-reported diff {d}"
    return f"megakernel err={err:.1e}, self-reported diffs {report.max_abs_diff_vs_ref:.1e}/{rep2.max_abs_diff_vs_ref:.1e}"


def main() -> int:
    quick = "--quick" in sys.argv
    torch_ok = has_torch()
    print("Neuro-Physical Hierarchical MoA Backbone — smoke test")
    print(f"numpy {np.__version__}" + (f", torch available" if torch_ok else ", torch MISSING (torch checks skipped)"))
    print("-" * 72)

    check("imports (12 modules)", test_imports)
    check("numpy branch RMSNorm fwd/bwd + FD gradient", test_numpy_rmsnorm)
    check("numpy 4 SSM scans vs sequential reference", test_numpy_ssm_scans)
    check("numpy per-head MoA attention mask", test_numpy_attention_mask)
    check("numpy sparse top-k router + cross-proposal attention", test_numpy_router_and_attention)

    if not quick and torch_ok:
        check("torch HeterogeneousProposerOp fwd/bwd", test_torch_proposer)
        check("torch MoA stack (attention + 2 aggregators)", test_torch_moa_stack)
        check("torch domain operators (neuro, tidal)", test_domain_ops)
        check("triton tile simulators vs references", test_triton_simulators)
    elif not quick:
        print("  SKIP  torch / simulator sections (torch not installed)")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("-" * 72)
    print(f"{passed}/{total} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
