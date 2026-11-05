#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六阶十一域混合架构 · 机制动画（暗色数据报告风）

风格基准：moe_mechanism_full.mp4 —— 近黑底、左上标题、每章一行场景说明、
主体全部是真实数据图表（热图/柱状/折线）、等宽琥珀色标公式、底部结论横幅、
最底下章节进度条。无装饰图形。

数据来源：全部来自 hybrid_moa_vl_operator_suite 的真实前向/反向/计时，
不使用任何随机生成的假数据。有 CUDA 时用 T4 实跑 d_model=256/L=128/B=4，
并追加每阶段 GPU 时延；无 CUDA 时退到小配置（仅用于本地开发调试）。

用法：
    python3 build_v6_film.py capture     # 采数据 -> data.npz
    python3 build_v6_film.py render      # 出帧 + 合成 -> v6_mechanism.mp4
    python3 build_v6_film.py all         # 两步都做
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)                                    # import hybrid_moa_vl_operator_suite
sys.path.insert(0, os.path.join(REPO, "hybrid_moa_vl_operator_suite"))  # 平目录 import
DATA = os.path.join(HERE, "data.npz")
OUTDIR = os.path.join(HERE, "out")
os.makedirs(OUTDIR, exist_ok=True)

GPU = False
try:
    import torch

    GPU = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    torch = None

# ---------------------------------------------------------------- 配置 -------
if GPU:
    CFG = dict(d_model=256, n_heads=8, num_proposers=3, top_k_proposers=2,
               num_vacuum_modes=64, num_cross_modal_anchors=64, num_local_groups=16,
               local_rank=16, num_extra_dims=7, num_tulpas=4, num_dream_anchors=64,
               n_layers=2)
    B, L = 4, 128
else:  # 本地开发用的小配置
    CFG = dict(d_model=64, n_heads=4, num_proposers=3, top_k_proposers=2,
               num_vacuum_modes=16, num_cross_modal_anchors=16, num_local_groups=4,
               local_rank=2, num_extra_dims=3, num_tulpas=2, num_dream_anchors=16,
               n_layers=1)
    B, L = 2, 24
D = CFG["d_model"]

# 12 个领域算子（顺序即片中编号）
OPS = [
    ("VacuumZPE_Fluctuation", "vacuum_zpe_fluctuation_operators",
     "VacuumZPEFluctuationConfig", "VacuumZPEFluctuationOp"),
    ("Parity_Baryogenesis", "baryogenesis_parity_operators",
     "ParityBaryogenesisConfig", "ParityBaryogenesisAsymmetryOp"),
    ("Supermembrane_11D", "supermembrane_spacetime_operators",
     "SupermembraneSpacetimeConfig", "SupermembraneSpacetimeOp"),
    ("QuantumTidal_WKB", "quantum_tidal_operators",
     "QuantumTidalConfig", "QuantumTidalFluctuationOp"),
    ("SpinalCord_Pathway", "spinal_cord_pathway_operators",
     "SpinalCordPathwayConfig", "SpinalCordPathwayOp"),
    ("Corticospinal_Chaos", "corticospinal_chaotic_operators",
     "CorticospinalChaoticConfig", "CorticospinalChaoticPathwayOp"),
    ("Dissociative_Trauma", "dissociative_trauma_neuro_operators",
     "DissociativeTraumaConfig", "DissociativeTraumaNeuroOp"),
    ("CrossModal_VL_Fusion", "moa_operators", None, "CrossModalVisionLanguageFusionSubOp"),
    ("Cortical_Dendritic_NMDA", "neuro_nonlinear_operators",
     "NeuroNonlinearConfig", "NonlinearDendriticNeuroOp"),
    ("Synaptic_DeltaOja", "local_self_modifying_operators",
     "LocalSelfModConfig", "LocalNonlinearSelfModifyingOp"),
    ("Tulpa_Persona_Agency", "tulpa_neuro_operators",
     "TulpaNeuroConfig", "TulpaNeuroCognitiveOp"),
    ("Sleep_Subconscious", "sleep_subconscious_neuro_operators",
     "SleepSubconsciousConfig", "SleepSubconsciousNeuroOp"),
]


def _cfg_kwargs(mod, cfg_name):
    """只传该 Config 真正接受的字段，避免不同算子签名不一致。"""
    if cfg_name is None:
        return {}
    flds = mod.__dict__[cfg_name].__dataclass_fields__
    kw = {"d_model": D, "n_heads": CFG["n_heads"]}
    kw = {k: v for k, v in kw.items() if k in flds}
    for key, cfgkey in [("num_vacuum_modes", "num_vacuum_modes"),
                        ("num_dream_anchors", "num_dream_anchors"),
                        ("num_local_groups", "num_local_groups"),
                        ("local_rank", "local_rank"),
                        ("num_extra_dims", "num_extra_dims"),
                        ("num_tulpas", "num_tulpas")]:
        if cfgkey in flds:
            kw[key] = CFG[cfgkey]
    return kw


def build_op(idx, dev="cpu"):
    """按编号实例化第 idx 个领域算子（0-based）。"""
    import importlib

    name, mod_name, cfg_name, op_name = OPS[idx]
    mod = importlib.import_module(mod_name)
    if cfg_name is None:  # 跨模态融合：直接吃 d_model / num_anchors
        return getattr(mod, op_name)(d_model=D, num_anchors=CFG["num_cross_modal_anchors"])
    op = getattr(mod, op_name)(getattr(mod, cfg_name)(**_cfg_kwargs(mod, cfg_name)))
    return op.to(dev) if hasattr(op, "to") else op


# ============================================================== 数据采集 ======
def capture():
    import hybrid_moa_vl_operator_suite as P
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
    keep = ["zpe_out", "baryon_out", "brane_out", "qtf_out", "spinal_out",
            "chaotic_bridge_out", "trauma_out", "g_shut_gate", "reflex_bypass",
            "proposals_raw", "router_logits", "active_weights", "f_i",
            "h_cortex", "tulpa_out", "h_wake", "y_consolidated", "mega_syn"]
    for k in keep:
        out[f"np_{k}"] = np.asarray(ctx[k])
    out["np_x_in"] = x_np
    out["np_x_zpe"] = np.asarray(ctx["x_zpe"])
    out["np_x_matter"] = np.asarray(ctx["x_matter"])
    out["np_x_spacetime"] = np.asarray(ctx["x_spacetime"])

    # ---- 1b) 模块级解析内核直采：真空锚点 / Bogoliubov / 梦境锚点 / 类淋巴 ----
    from sleep_subconscious_neuro_operators import SleepSubconsciousAnalyticalKernels as SK
    from vacuum_zpe_fluctuation_operators import VacuumZPEAnalyticalKernels as VK

    rng2 = np.random.default_rng(7)
    zc, sc = ctx["zpe_cache"], ctx["sleep_cache"]
    out["zpe_q"] = np.asarray(zc["q"])
    out["zpe_p"] = np.asarray(zc["p"])
    out["zpe_omega"] = np.asarray(zc["omega_zpe"])
    out["zpe_r_squeeze"] = np.asarray(params["r_squeeze"])
    out["zpe_eps_floor"] = np.array([0.05])  # VacuumZPEFluctuationConfig 默认零点能下限
    vmodes = rng2.standard_normal((CFG["num_vacuum_modes"], D)) * (D ** -0.5)
    _, vcache = VK.vacuum_virtual_boson_propagator_forward(zc["q"], zc["p"], vmodes)
    out["vac_attn_emit"] = np.asarray(vcache["attn_emit"])   # [B, R, L]
    out["vac_attn_abs"] = np.asarray(vcache["attn_abs"])     # [B, L, R]
    out["vac_v_boson"] = np.asarray(vcache["v_boson"])       # [B, R, D]

    out["sleep_w_glymph"] = np.asarray(sc["w_glymph"])
    out["sleep_x_wake"] = np.asarray(sc["x_wake"])
    out["sleep_inv_shy"] = np.asarray(sc["inv_shy"])
    out["sleep_g"] = np.asarray(sc["g_sleep"])
    danchors = rng2.standard_normal((CFG["num_dream_anchors"], D)) * (D ** -0.5)
    _, dcache = SK.subconscious_pgo_dream_recombination_forward(
        sc["x_wake"], sc["m_hipp"], danchors)
    out["dream_attn_pool"] = np.asarray(dcache["attn_pool"])  # [B, R, L]
    out["dream_attn_bc"] = np.asarray(dcache["attn_bc"])      # [B, L, R]

    # KK 模态塔（Stage I.A 的紧致化模态）
    bc = ctx["brane_cache"]
    out["kk_w"] = np.asarray(bc["w_kk"])
    out["kk_omega"] = np.asarray(bc["omega_kk"])
    out["kk_kappa"] = np.asarray(bc["kappa_warp"])

    # ---- 1c) 有界性曲线：输入幅度扫描 -> 输出幅度（真测，不是理论声明）----
    def _bound_curve(op_idx, scales=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)):
        import importlib

        mod = importlib.import_module(OPS[op_idx][1])
        cfg_name, op_name = OPS[op_idx][2], OPS[op_idx][3]
        op = getattr(mod, op_name)(getattr(mod, cfg_name)(**_cfg_kwargs(mod, cfg_name)))
        is_torch = torch is not None and isinstance(op, torch.nn.Module)
        ins, outs = [], []
        for s in scales:
            xs = rng2.standard_normal((1, L, D)) * s
            xin = torch.from_numpy(xs).float() if is_torch else xs
            r = op(xin)
            y = r[0] if isinstance(r, tuple) else r
            yv = float(torch.linalg.norm(y).item()) if is_torch else float(np.linalg.norm(np.asarray(y)))
            ins.append(float(np.linalg.norm(xs)))
            outs.append(yv)
        return np.array(ins), np.array(outs)

    for idx, tag in [(0, "vac"), (5, "chaos"), (6, "trauma"), (11, "sleep")]:
        i_, o_ = _bound_curve(idx)
        out[f"bound_in_{tag}"] = i_
        out[f"bound_out_{tag}"] = o_
    out["bound_scales"] = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

    # 常态 vs +50 冲击（互补门控）
    x_shock = x_np.copy()
    x_shock[:, 0, :] += 50.0
    y_s, ctx_s = UnifiedArchitectureExecutionEngine.forward_unified_layer_exact(
        x_shock, params, cfg, use_smooth_router=True)
    out["np_g_shut_shock"] = np.asarray(ctx_s["g_shut_gate"])
    out["np_y_consolidated_shock"] = np.asarray(ctx_s["y_consolidated"])
    out["np_trauma_shock"] = np.asarray(ctx_s["trauma_out"])

    # ---- 2) torch 层：14 个阶段标量诊断 + 真实反向 ----
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
        out["torch_grad"] = xt.grad.detach().cpu().numpy()
        out["torch_params"] = float(sum(p.numel() for p in layer.parameters()))

        # ---- 3) 每阶段 GPU/CPU 时延 + 参数量 ----
        lat, par, names = [], [], []
        xin = torch.randn(B, L, D, device=dev)
        for i, (name, *_rest) in enumerate(OPS):
            op = build_op(i, dev)
            n_par = sum(p.numel() for p in op.parameters()) if hasattr(op, "parameters") else 0
            # 预热
            try:
                r = op(xin)
                if isinstance(r, tuple):
                    r[0].float().pow(2).mean().backward()
            except Exception as e:  # noqa: BLE001
                print(f"    [timing] {name} 跳过：{e}")
                continue
            if GPU:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 5
            for _ in range(reps):
                xin2 = xin.clone().requires_grad_(True)
                r = op(xin2)
                loss = r[0] if isinstance(r, tuple) else r
                loss.float().pow(2).mean().backward()
            if GPU:
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) / reps * 1000.0)
            par.append(float(n_par))
            names.append(name)
        out["timing_names"] = np.array(names)
        out["timing_ms"] = np.array(lat)
        out["timing_params"] = np.array(par)
        if GPU:
            out["gpu_name"] = np.array([torch.cuda.get_device_name(0)])
            out["gpu_mem_gb"] = np.array([torch.cuda.max_memory_allocated() / 1e9])

        # ---- 4) 12×12 参数交叉灵敏雅可比 ----
        ops = [build_op(i, dev) for i in range(len(OPS))]
        base = []
        with torch.no_grad():
            for op in ops:
                r = op(xin)
                base.append((r[0] if isinstance(r, tuple) else r).clone())
        J = np.zeros((len(OPS), len(OPS)))
        delta = 1e-3
        for j, op in enumerate(ops):
            ps = [p for p in op.parameters()] if hasattr(op, "parameters") else []
            if not ps:
                continue
            with torch.no_grad():
                snap = [p.detach().clone() for p in ps]
                for p in ps:
                    p.add_(delta)
                for i, opi in enumerate(ops):
                    r = opi(xin)
                    yi = r[0] if isinstance(r, tuple) else r
                    J[i, j] = float((yi - base[i]).abs().max().cpu())
                for p, s in zip(ps, snap):
                    p.copy_(s)
        out["jacobian"] = J

    np.savez_compressed(DATA, **out)
    print(f"[capture] saved {DATA}  ({len(out)} arrays, GPU={GPU})")
    for k in ["diag_stage5_vlpo_sleep_gate", "timing_ms", "jacobian"]:
        if k in out:
            print(f"    {k}: {np.asarray(out[k]).shape}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("capture", "all"):
        capture()
    if cmd in ("render", "all"):
        import render_film

        render_film.main()
