#!/usr/bin/env python3
"""病毒核酸侵入算子验收测试。

覆盖 7 项：
  1. 构造与形状
  2. 状态单纯形约束（7 个潜伏占比和为 1）
  3. 门控：integrated=0 → 表达严格为 0
  4. 沿 L 因果（第 t 个输出不依赖 t' > t）
  5. 样本独立（fp64 下必须逐位相同）
  6. 确定性 + 两后端一致
  7. 13 条 PMID 机制约束：人为破坏必须被 MechanismViolation 拦截

运行：
    python3 tests/plugin_acceptance/test_viral_invasion.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from hybrid_moa_vl_operator_suite.viral_invasion_operator import (  # noqa: E402
    CoupledViralInvasionConfig, MechanismViolation, create_viral_invasion_operator)

MOD_PATH = os.path.abspath(os.path.join(
    _HERE, "..", "..", "hybrid_moa_vl_operator_suite", "viral_invasion_operator.py"))


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name}  {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    results = []
    cfg = CoupledViralInvasionConfig(d_model=16)

    def t1():
        op = create_viral_invasion_operator(cfg, backend="torch")
        x = torch.randn(2, 8, 16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, info = op(x, return_aux=True)
        assert y.shape == x.shape
        st = info["state_trajectory"]
        assert st.shape[-1] == 7, st.shape
        return f"y={tuple(y.shape)}, 状态轨迹={tuple(st.shape)}"

    results.append(check("1. 构造与形状", t1))

    def t2():
        op = create_viral_invasion_operator(cfg, backend="numpy")
        x = torch.randn(2, 8, 16).numpy()
        _, info = op(x, return_aux=True)
        s = info["state_trajectory"].sum(-1)
        d = float(abs(s - 1.0).max())
        assert d < 1e-12, d
        assert float(info["state_trajectory"].min()) >= 0.0
        return f"状态和 max|Σ-1| = {d:.2e}"

    results.append(check("2. 单纯形约束", t2))

    def t3():
        import numpy as np
        # 直接测内部：integrated=0 时 expression 的有效速率必须为 0
        mod = sys.modules["hybrid_moa_vl_operator_suite.viral_invasion_operator"]
        fb, gm = mod._topology(), mod._gate_mask()
        s = np.full((1, 1, 7), 0.05)
        s[0, 0, 0] = 1 - 6 * 0.05
        s[0, 0, mod._GATE_STATE] = 0.0
        rates, gate = mod._transition_rates(s, np.zeros((1, 1, 11)), fb, np)
        gate_v = float(gate[0, 0, 0])
        assert gate_v == 0.0, gate_v
        eff = float(rates[0, 0, mod._GATED_EDGE]) * (1.0 + 1.0 * (gate_v - 1.0))
        assert abs(eff) < 1e-15, eff
        return f"gate={gate_v:.1e}，expression 有效速率={eff:.1e}"

    results.append(check("3. 门控 integrated=0 → 表达=0", t3))

    def t4():
        op = create_viral_invasion_operator(cfg, backend="torch")
        torch.manual_seed(0)
        xa = torch.randn(1, 8, 16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ya, _ = op(xa, return_aux=True)
            xb = xa.clone()
            xb[0, 5:] += 3.0
            yb, _ = op(xb, return_aux=True)
        d = float((ya[0, :5].detach() - yb[0, :5].detach()).abs().max())
        assert d < 1e-12, d
        return f"改 t>=5 后 t<5 输出 max|Δ| = {d:.2e}"

    results.append(check("4. 沿 L 因果", t4))

    def t5():
        op = create_viral_invasion_operator(cfg, backend="torch", dtype=torch.float64)
        torch.manual_seed(0)
        x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = op(x, return_aux=True)
            gb = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
            gs = []
            for i in range(2):
                xi = x[i:i + 1].detach().requires_grad_(True)
                op(xi, return_aux=True)[0].sum().backward()
                gs.append(xi.grad)
        d = float((gb - torch.cat(gs, 0)).abs().max())
        # 必须用 .sum() 而非 .mean()：后者会再对 batch 平均一次，造成 1/B 的假差异
        assert d == 0.0, d
        return "fp64 下 batch 梯度与逐样本梯度逐位相同"

    results.append(check("5. 样本独立（fp64 逐位）", t5))

    def t6():
        op = create_viral_invasion_operator(cfg, backend="torch")
        torch.manual_seed(0)
        x = torch.randn(1, 8, 16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1, _ = op(x, return_aux=True)
            y2, _ = op(x, return_aux=True)
        assert torch.equal(y1, y2)
        on = create_viral_invasion_operator(cfg, backend="numpy")
        xn = x.numpy()
        dn = float(abs(on(xn) - y1.detach().numpy()).max())
        assert dn < 1e-6, dn
        return f"确定性 OK；两后端 max|Δ| = {dn:.2e}"

    results.append(check("6. 确定性与两后端一致", t6))

    def t7():
        src = open(MOD_PATH, encoding="utf-8").read()
        mutations = [
            ("dna_sensing 目标改成 expressed",
             "_EDGE_TARGET = (1, 2, 3, 4, 5, 6, 6, 0, 0, 0, 3)",
             "_EDGE_TARGET = (1, 2, 3, 4, 5, 4, 6, 0, 0, 0, 3)"),
            ("IFN→entry 符号翻转",
             "feedback[6, 0] = -0.9", "feedback[6, 0] = +0.9"),
            ("integrated 加第二条出边",
             "_EDGE_SOURCE = (0, 1, 2, 0, 1, 2, 5, 6, 3, 4, 5)",
             "_EDGE_SOURCE = (0, 1, 2, 0, 1, 2, 5, 6, 3, 4, 3)"),
        ]
        caught = 0
        for label, old, new in mutations:
            assert old in src, f"变异点未找到: {label}"
            mutated = src.replace(old, new, 1)
            d = tempfile.mkdtemp()
            p = os.path.join(d, "mut_mod.py")
            open(p, "w", encoding="utf-8").write(mutated)
            try:
                r = subprocess.run([sys.executable, "-c",
                                    f"import sys; sys.path.insert(0,'{d}');"
                                    "import mut_mod; mut_mod._topology()"],
                                   capture_output=True, text=True, timeout=120)
                if "MechanismViolation" in (r.stderr + r.stdout):
                    caught += 1
            finally:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
        assert caught == len(mutations), f"只拦住 {caught}/{len(mutations)}"
        return f"{caught}/{len(mutations)} 种人为破坏被 MechanismViolation 拦截"

    results.append(check("7. 机制约束真的会拦人", t7))

    print(f"\n{sum(results)}/{len(results)} 通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
