#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU launch 回归检查：把 14 个领域算子逐个在 CUDA 上跑前向 + 反向。

存在的理由：v12 的包里有一类「Triton launch 传参与内核签名不一致」的缺陷
（baryogenesis 的 delta_mass_sq、neuro 的 kappa_sq3、quantum_tidal 的 M_HARMONICS、
supermembrane 的 tau_sq/beta_sq）。这类缺陷在 CPU（numpy 路径）上完全测不出来，
必须真跑 CUDA 才能发现。

用法：
    python3 check_gpu_launch.py            # 全 14 个算子 fwd+bwd + 与 numpy 路径对拍
    python3 check_gpu_launch.py --quick    # 只跑 fwd+bwd，不对拍
退出码 0 = 全过。
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hybrid_moa_vl_operator_suite"))

OPS = [
    ("VacuumZPE", "vacuum_zpe_fluctuation_operators", "VacuumZPEFluctuationConfig",
     "VacuumZPEFluctuationOp"),
    ("ParityBaryogenesis", "baryogenesis_parity_operators", "ParityBaryogenesisConfig",
     "ParityBaryogenesisAsymmetryOp"),
    ("Supermembrane", "supermembrane_spacetime_operators", "SupermembraneSpacetimeConfig",
     "SupermembraneSpacetimeOp"),
    ("QuantumTidal", "quantum_tidal_operators", "QuantumTidalConfig", "QuantumTidalFluctuationOp"),
    ("QuantumResource", "quantum_resource_phasing_operators", "QuantumResourcePhasingConfig",
     "QuantumResourcePhasingOp"),
    ("SpinalCord", "spinal_cord_pathway_operators", "SpinalCordPathwayConfig",
     "SpinalCordPathwayOp"),
    ("CorticospinalChaos", "corticospinal_chaotic_operators", "CorticospinalChaoticConfig",
     "CorticospinalChaoticPathwayOp"),
    ("DissociativeTrauma", "dissociative_trauma_neuro_operators", "DissociativeTraumaConfig",
     "DissociativeTraumaNeuroOp"),
    ("CrossModalVL", "moa_operators", None, "CrossModalVisionLanguageFusionSubOp"),
    ("CorticalDendritic", "neuro_nonlinear_operators", "NeuroNonlinearConfig",
     "NonlinearDendriticNeuroOp"),
    ("SynapticDeltaOja", "local_self_modifying_operators", "LocalSelfModConfig",
     "LocalNonlinearSelfModifyingOp"),
    ("Tulpa", "tulpa_neuro_operators", "TulpaNeuroConfig", "TulpaNeuroCognitiveOp"),
    ("Sleep", "sleep_subconscious_neuro_operators", "SleepSubconsciousConfig",
     "SleepSubconsciousNeuroOp"),
    ("CPUGPU", "cpu_gpu_self_bootstrapping_operators", "CPUGPUSelfBootstrappingConfig",
     "CPUGPUSelfBootstrappingOp"),
]
CFGK = dict(d_model=64, n_heads=4, num_vacuum_modes=16, num_dream_anchors=16,
            num_isa_anchors=16, num_local_groups=4, local_rank=2, num_extra_dims=3,
            num_tulpas=2)


def cfg_kwargs(mod, cfg_name):
    if cfg_name is None:
        return {}
    flds = mod.__dict__[cfg_name].__dataclass_fields__
    kw = {k: v for k, v in CFGK.items() if k in flds}
    return kw


def build(idx, dev):
    import importlib

    _n, mod_name, cfg_name, op_name = OPS[idx]
    mod = importlib.import_module(mod_name)
    if cfg_name is None:
        op = getattr(mod, op_name)(d_model=CFGK["d_model"], num_anchors=16)
    else:
        op = getattr(mod, op_name)(getattr(mod, cfg_name)(**cfg_kwargs(mod, cfg_name)))
    return op.to(dev) if hasattr(op, "to") else op


def main():
    import torch

    if not torch.cuda.is_available():
        print("需要 CUDA")
        return 2
    dev = "cuda"
    B, L, D = 2, 24, 64
    quick = "--quick" in sys.argv
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"逐算子 CUDA 前向 + 反向检查（{len(OPS)} 个）")
    ok = 0
    fails = []
    for i, (name, *_r) in enumerate(OPS):
        try:
            op = build(i, dev)
            x = torch.randn(B, L, D, device=dev, requires_grad=True)
            r = op(x)
            out = r[0] if isinstance(r, tuple) else r
            out.float().pow(2).mean().backward()
            g = float(x.grad.norm().cpu())
            assert torch.isfinite(out).all(), "输出非有限"
            assert g > 0.0, "梯度为零"
            extra = ""
            if not quick:
                # 与 numpy/CPU 路径对拍：同一权重、同一输入
                op_cpu = build(i, "cpu")
                op_cpu.load_state_dict({k: v.detach().cpu() for k, v in op.state_dict().items()})
                xc = x.detach().cpu().clone().requires_grad_(True)
                rc = op_cpu(xc)
                outc = rc[0] if isinstance(rc, tuple) else rc
                d = float((out.detach().cpu() - outc.detach()).abs().max())
                rel = d / max(float(outc.detach().abs().max()), 1e-9)
                extra = f" | 与 CPU 路径 max|diff|={d:.2e} (rel {rel:.1e})"
                assert rel < 1e-4, f"CPU/GPU 结果不一致 rel={rel:.2e}"
            ok += 1
            print(f"  PASS  {name:22s} out={tuple(out.shape)} |grad|={g:.4f}{extra}")
        except Exception as exc:  # noqa: BLE001
            fails.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name:22s} {type(exc).__name__}: {str(exc)[:160]}")
            traceback.print_exc(limit=2)
    print("-" * 70)
    print(f"{ok}/{len(OPS)} 算子 CUDA 前向+反向通过")
    for n, e in fails:
        print(f"  FAILED {n}: {e}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
