#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13 域 14 算子 7 阶段 4 地层 · 机制动画 —— 数据采集

数据全部来自真实前向 / 反向 / 计时，无随机生成的假数据。
有 CUDA 时自动切 T4 真实尺度（d_model=256 / L=128 / B=4），并追加逐算子时延；
无 CUDA 时退到小配置（仅本地开发调试）。

用法：
    python3 build_film.py capture     # 采数据 -> data.npz
    python3 build_film.py render      # 出帧 + 合成 -> v12_mechanism.mp4
    python3 build_film.py all
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hybrid_moa_vl_operator_suite"))
DATA = os.path.join(HERE, "data.npz")
os.makedirs(HERE, exist_ok=True)

GPU = False
try:
    import torch

    GPU = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    torch = None

if GPU:
    CFG = dict(d_model=256, n_heads=8, n_layers=2, num_proposers=3, top_k_proposers=2,
               num_vacuum_modes=64, num_cross_modal_anchors=64, num_local_groups=16,
               local_rank=16, num_isa_anchors=64, num_extra_dims=7, num_tulpas=4,
               num_dream_anchors=64)
    B, L = 4, 128
else:
    CFG = dict(d_model=64, n_heads=4, n_layers=1, num_proposers=3, top_k_proposers=2,
               num_vacuum_modes=16, num_cross_modal_anchors=16, num_local_groups=4,
               local_rank=2, num_isa_anchors=16, num_extra_dims=3, num_tulpas=2,
               num_dream_anchors=16)
    B, L = 2, 24
D = CFG["d_model"]

# 14 个领域算子（顺序 = 片中编号，与五维审计的 DOMAIN_OPERATOR_NAMES 一致）
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


def _cfg_kwargs(mod, cfg_name):
    """只传该 Config 真正接受的字段。"""
    if cfg_name is None:
        return {}
    flds = mod.__dict__[cfg_name].__dataclass_fields__
    kw = {"d_model": D, "n_heads": CFG["n_heads"]}
    kw = {k: v for k, v in kw.items() if k in flds}
    for key in ("num_vacuum_modes", "num_dream_anchors", "num_isa_anchors",
                "num_local_groups", "local_rank", "num_extra_dims", "num_tulpas"):
        if key in flds:
            kw[key] = CFG[key]
    return kw


def build_op(idx, dev="cpu"):
    import importlib

    _name, mod_name, cfg_name, op_name = OPS[idx]
    mod = importlib.import_module(mod_name)
    if cfg_name is None:
        op = getattr(mod, op_name)(d_model=D, num_anchors=CFG["num_cross_modal_anchors"])
    else:
        op = getattr(mod, op_name)(getattr(mod, cfg_name)(**_cfg_kwargs(mod, cfg_name)))
    return op.to(dev) if hasattr(op, "to") else op


# ============================================================== 数据采集 ======
def capture():
    from hybrid_moa_vl_operator_suite.unified_hybrid_moa_architecture import (
        UnifiedArchitectureExecutionEngine, UnifiedHybridArchitectureConfig,
        UnifiedInterdisciplinaryMoALayer)

    rng = np.random.default_rng(11)
    x_np = (rng.standard_normal((B, L, D)) * 0.5).astype(np.float64)
    cfg = UnifiedHybridArchitectureConfig(**CFG)
    params = UnifiedArchitectureExecutionEngine.init_layer_parameters(D, cfg, seed=2026)
    out = {}

    # ---- 1) numpy 引擎：逐阶段张量 ----
    y, ctx = UnifiedArchitectureExecutionEngine.forward_unified_layer_exact(
        x_np, params, cfg, use_smooth_router=True)
    keep = ["zpe_out", "baryon_out", "brane_out", "qtf_out", "qrp_out", "spinal_out",
            "chaotic_bridge_out", "trauma_out", "g_shut_gate", "reflex_bypass",
            "proposals_raw", "router_logits", "active_weights", "f_i", "h_cortex",
            "tulpa_out", "h_wake", "y_consolidated", "mega_syn", "boot_out",
            "w_hist_oja", "x_tidal", "x_spacetime", "x_zpe", "x_matter", "q_oja",
            "k_oja", "v_oja", "eta_gate_oja", "gam_gate_oja", "h_silicon"]
    for k in keep:
        if k in ctx:
            out[f"np_{k}"] = np.asarray(ctx[k])
    out["np_x_in"] = x_np

    # 常态 vs +50 冲击（互补门控）
    x_shock = x_np.copy()
    x_shock[:, 0, :] += 50.0
    _, ctx_s = UnifiedArchitectureExecutionEngine.forward_unified_layer_exact(
        x_shock, params, cfg, use_smooth_router=True)
    out["np_g_shut_shock"] = np.asarray(ctx_s["g_shut_gate"])
    out["np_y_consolidated_shock"] = np.asarray(ctx_s["y_consolidated"])
    out["np_trauma_shock"] = np.asarray(ctx_s["trauma_out"])

    # ---- 2) torch 层：21 项阶段遥测 + 真实 backward ----
    if torch is not None:
        dev = "cuda" if GPU else "cpu"
        torch.manual_seed(0)
        layer = UnifiedInterdisciplinaryMoALayer(cfg).to(dev)
        xt = torch.randn(B, L, D, device=dev, requires_grad=True)
        y_t, aux, diag = layer(xt)
        for k, v in diag.items():
            try:
                out[f"diag_{k}"] = float(np.asarray(
                    v.detach().cpu() if torch.is_tensor(v) else v))
            except Exception:  # noqa: BLE001
                pass
        y_t.float().pow(2).mean().backward()
        out["torch_grad_norm"] = float(xt.grad.norm().cpu())
        out["torch_act"] = y_t.detach().cpu().numpy()
        out["torch_params"] = float(sum(p.numel() for p in layer.parameters()))
        if GPU:
            out["gpu_name"] = np.array([torch.cuda.get_device_name(0)])
            out["gpu_mem_gb"] = np.array([torch.cuda.max_memory_allocated() / 1e9])

    # ---- 3) 每算子前向+反向时延（GPU 真实尺度） ----
    lat, par, names = [], [], []
    if torch is not None:
        dev = "cuda" if GPU else "cpu"
        xin = torch.randn(B, L, D, device=dev)
        for i, (name, *_r) in enumerate(OPS):
            try:
                op = build_op(i, dev)
            except Exception as e:  # noqa: BLE001
                print(f"    [timing] {name} 构造失败：{e}", flush=True)
                continue
            n_par = sum(p.numel() for p in op.parameters()) if hasattr(op, "parameters") else 0
            try:
                for _ in range(3):
                    t = xin.clone().requires_grad_(True)
                    r = op(t)
                    (r[0] if isinstance(r, tuple) else r).float().pow(2).mean().backward()
                if GPU:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                reps = 5
                for _ in range(reps):
                    t = xin.clone().requires_grad_(True)
                    r = op(t)
                    (r[0] if isinstance(r, tuple) else r).float().pow(2).mean().backward()
                if GPU:
                    torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) / reps * 1000.0)
                par.append(float(n_par))
                names.append(name)
            except Exception as e:  # noqa: BLE001
                print(f"    [timing] {name} 跳过：{e}", flush=True)
        out["timing_names"] = np.array(names)
        out["timing_ms"] = np.array(lat)
        out["timing_params"] = np.array(par)

    # ---- 4) 14×14 参数交叉灵敏雅可比（复用上游审计同一函数，口径一致） ----
    try:
        from operator_independence_verification import (
            evaluate_all_14_domain_operators_independently as eval_ops)

        xs = np.random.default_rng(2026).standard_normal((8, 24, 32)) * 0.7
        gs = np.random.default_rng(7).standard_normal((8, 24, 32))
        base, _ = eval_ops(xs, gs, 2026, -1)
        J = np.zeros((len(base), len(base)))
        for j in range(len(base)):
            oj, _ = eval_ops(xs, gs, 2026, j)
            for i in range(len(base)):
                J[i, j] = float(np.max(np.abs(oj[i] - base[i])))
        out["jacobian"] = J
        out["jacobian_names"] = np.array([n for n, *_r in OPS][:J.shape[0]])
    except Exception as e:  # noqa: BLE001
        print("    [jacobian] 跳过：", e, flush=True)

    # ---- 4b) 模块级解析内核直采：真空锚点共注意力 / 梦境锚点 / 有界性曲线 ----
    try:
        from sleep_subconscious_neuro_operators import SleepSubconsciousAnalyticalKernels as SK
        from vacuum_zpe_fluctuation_operators import VacuumZPEAnalyticalKernels as VK

        zc, sc = ctx.get("zpe_cache", {}), ctx.get("sleep_cache", {})
        if zc:
            out["np_zpe_q"] = np.asarray(zc["q"])
            out["np_zpe_p"] = np.asarray(zc["p"])
            out["np_zpe_omega"] = np.asarray(zc["omega_zpe"])
            out["np_zpe_r_squeeze"] = np.asarray(zc["r_squeeze"])
            modes = np.random.default_rng(7).standard_normal((CFG["num_vacuum_modes"], D)) * (D ** -0.5)
            _, vc = VK.vacuum_virtual_boson_propagator_forward(
                zc["q"], zc["p"], modes)
            out["vac_attn_emit"] = np.asarray(vc["attn_emit"])       # [B, R, L]
            out["vac_attn_abs"] = np.asarray(vc["attn_abs"])         # [B, L, R]
        if sc:
            out["np_sleep_x_wake"] = np.asarray(sc["x_wake"])
            out["np_sleep_w_glymph"] = np.asarray(sc["w_glymph"])
            out["np_sleep_inv_shy"] = np.asarray(sc["inv_shy"])
            out["np_sleep_g"] = np.asarray(sc["g_sleep"])
            out["np_sleep_d_shy"] = np.asarray(sc["d_shy"])
            danchors = np.random.default_rng(7).standard_normal((CFG["num_dream_anchors"], D)) * (D ** -0.5)
            _, dc = SK.subconscious_pgo_dream_recombination_forward(
                sc["x_wake"], sc["m_hipp"], danchors)
            out["dream_attn_pool"] = np.asarray(dc["attn_pool"])     # [B, R, L]
            out["dream_attn_bc"] = np.asarray(dc["attn_bc"])         # [B, L, R]
    except Exception as e:  # noqa: BLE001
        print("    [anchors] 跳过：", e, flush=True)

    def _bound(op_idx, scales=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)):
        import importlib

        _n, mod_name, cfg_name, op_name = OPS[op_idx]
        mod = importlib.import_module(mod_name)
        if cfg_name is None:
            op = getattr(mod, op_name)(d_model=D, num_anchors=CFG["num_cross_modal_anchors"])
        else:
            op = getattr(mod, op_name)(getattr(mod, cfg_name)(**_cfg_kwargs(mod, cfg_name)))
        is_torch = torch is not None and isinstance(op, torch.nn.Module)
        ins, outs = [], []
        for sc_ in scales:
            xs_ = rng.standard_normal((1, L, D)) * sc_
            xin_ = torch.from_numpy(xs_).float() if is_torch else xs_
            r = op(xin_)
            yv = r[0] if isinstance(r, tuple) else r
            outs.append(float(torch.linalg.norm(yv).item()) if is_torch
                       else float(np.linalg.norm(np.asarray(yv))))
            ins.append(float(np.linalg.norm(xs_)))
        return np.array(ins), np.array(outs)

    for idx, tag in [(0, "vac"), (6, "chaos"), (7, "trauma"), (12, "sleep")]:
        try:
            i_, o_ = _bound(idx)
            out[f"bound_in_{tag}"] = i_
            out[f"bound_out_{tag}"] = o_
        except Exception as e:  # noqa: BLE001
            print(f"    [bound {tag}] 跳过：{e}", flush=True)

    # ---- 5) DeltaOja persistent kernel 的 launch 数（结构性事实） ----
    out["oja_launches_eager"] = np.array([L * CFG["num_local_groups"]])
    out["oja_launches_persistent"] = np.array([1])

    np.savez_compressed(DATA, **out)
    print(f"[capture] saved {DATA}  ({len(out)} arrays, GPU={GPU})")
    for k in ("timing_ms", "jacobian", "np_qrp_out"):
        if k in out:
            print(f"    {k}: {np.asarray(out[k]).shape}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("capture", "all"):
        capture()
    if cmd in ("render", "all"):
        import render_film

        render_film.main()
