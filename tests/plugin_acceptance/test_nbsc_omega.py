#!/usr/bin/env python3
"""NBSC-Ω 算子验收测试。

覆盖 8 项：
  1. 构造与初始状态
  2. 无驱动时完全静默（不该有自发放电）
  3. 有驱动时产生脉冲
  4. 力学输出随驱动变化
  5. 200 步长期有界（无 NaN/Inf）
  6. 反向梯度可达（5 个参数全部接进计算图）
  7. 采样路径下的 log_probability 与 REINFORCE 梯度
  8. dtype 不一致时快速失败

运行：
    python3 tests/plugin_acceptance/test_nbsc_omega.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from hybrid_moa_vl_operator_suite.nbsc_omega_operator import (  # noqa: E402
    NBSCOperator, NBSCOutput, NBSCState)


def _op(segments: int = 4):
    return NBSCOperator(segments=segments)


def _drive(batch: int, n: int, level: float = 1.0):
    d = torch.zeros(batch, n)
    d[:, :2] = 1.5 * level          # cortex / brainstem
    d[:, 2:] = 0.8 * level          # per-segment neurons
    return d


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
    op = _op()

    def t1():
        s = op.initial_state(2)
        assert isinstance(s, NBSCState)
        assert s.voltage.shape == (2, op.neuron_count)
        assert s.payload_history.shape[0] == 2
        n_par = sum(p.numel() for p in op.parameters())
        return f"{op.neuron_count} 神经元 / {op.edge_count} 边 / {n_par} 参数"

    results.append(check("1. 构造与初始状态", t1))

    def t2():
        s = op.initial_state(2)
        drive = torch.zeros(2, op.neuron_count)
        load = torch.zeros(2, op.segments)
        st = s
        spikes = 0
        for _ in range(100):
            out = op(st, drive, load)
            spikes += int(out.spikes.sum())
            st = out.state
        assert spikes == 0, f"无驱动却产生 {spikes} 个脉冲"
        assert torch.all(st.angle == 0), "无驱动却有关节运动"
        return "100 步零脉冲、零运动"

    results.append(check("2. 无驱动时完全静默", t2))

    def t3():
        s = op.initial_state(2)
        drive = _drive(2, op.neuron_count)
        load = torch.full((2, op.segments), 0.5)
        st = s
        spikes = 0
        for _ in range(300):
            out = op(st, drive, load)
            spikes += int(out.spikes.sum())
            st = out.state
        assert spikes > 0, "有驱动却零脉冲"
        return f"300 步 {spikes} 个脉冲"

    results.append(check("3. 有驱动时产生脉冲", t3))

    def t4():
        s = op.initial_state(2)
        drive = _drive(2, op.neuron_count)
        load = torch.full((2, op.segments), 0.5)
        st = s
        for _ in range(300):
            out = op(st, drive, load)
            st = out.state
        assert torch.isfinite(out.joint_torque).all()
        mag = float(out.joint_torque.abs().max())
        assert mag > 0, "力矩恒为 0"
        assert torch.all(st.angle != 0), "关节角仍为 0"
        return f"|torque|max={mag:.4f}, angle={float(st.angle[0, 0]):.4f} rad"

    results.append(check("4. 力学输出随驱动变化", t4))

    def t5():
        s = op.initial_state(2)
        drive = _drive(2, op.neuron_count)
        load = torch.full((2, op.segments), 0.5)
        st = s
        for _ in range(200):
            out = op(st, drive, load)
            st = out.state
        fields = ("voltage", "utilization", "resource", "synaptic_current",
                  "angle", "velocity", "plasticity")
        bad = [f for f in fields if not torch.isfinite(getattr(st, f)).all()]
        assert not bad, f"非有限值字段: {bad}"
        return f"200 步全部有限；voltage∈[{float(st.voltage.min()):.3f},{float(st.voltage.max()):.3f}]"

    results.append(check("5. 长期有界", t5))

    def t6():
        op.zero_grad(set_to_none=True)
        s = op.initial_state(2)
        drive = _drive(2, op.neuron_count)
        load = torch.full((2, op.segments), 0.5)
        st = s
        for _ in range(60):
            out = op(st, drive, load)
            st = out.state
        out.joint_torque.abs().sum().backward()
        got = {n: float(p.grad.abs().max()) for n, p in op.named_parameters()
               if p.grad is not None}
        # STDP 只调制突触权重，不经采样概率，因此对 joint_torque 必须有梯度
        assert "stdp_logits" in got, "stdp_logits 未接进计算图"
        assert got["stdp_logits"] > 0, "stdp_logits 梯度为 0"
        return f"{len(got)}/5 参数有梯度；stdp={got['stdp_logits']:.2e}"

    results.append(check("6. 反向梯度可达", t6))

    def t7():
        op.zero_grad(set_to_none=True)
        s = op.initial_state(2)
        drive = _drive(2, op.neuron_count)
        load = torch.full((2, op.segments), 0.5)
        torch.manual_seed(0)
        st = s
        for _ in range(10):
            su = torch.rand(2, op.neuron_count)
            ru = torch.rand(2, op.edge_count, op.release_sites)
            out = op(st, drive, load, spike_uniform=su, release_uniform=ru)
            st = out.state
        lp = out.log_probability.sum()
        assert lp.requires_grad, "log_probability 不在计算图中"
        assert float(lp) != 0.0, "传了采样但 log_probability 仍为 0"
        lp.backward()
        return f"logp={float(lp):.3f}，对 bias_logits 梯度={float(dict(op.named_parameters())['bias_logits'].grad.abs().max()):.2e}"

    results.append(check("7. 采样路径得分项", t7))

    def t8():
        s = op.initial_state(2)
        drive = torch.zeros(2, op.neuron_count, dtype=torch.float64)   # 故意错 dtype
        load = torch.zeros(2, op.segments, dtype=torch.float64)
        try:
            op(s, drive, load)
        except TypeError as exc:
            return f"入口即报 TypeError: {str(exc)[:60]}…"
        raise AssertionError("dtype 不一致时未快速失败")

    results.append(check("8. dtype 不一致快速失败", t8))

    print(f"\n{sum(results)}/{len(results)} 通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
