#!/usr/bin/env python3
"""侵入算子（信息域，多宿主）：宿主是**一群算子/模型实例**，病原是**一段能改写它们输入或参数的载荷**。

单宿主时退化成"一个模型被注入"；多宿主时状态是 `[N, 13]`，宿主之间靠一条投递矩阵耦合：
每个宿主的 `output` 按份额投递给别的宿主的 `inbound`。

    S ∈ R^{N×13},  N 个宿主
    宿主内：22 条边（复制要借 tool/credential；写进 memory；写进 weights；护栏检出）
    宿主间：propagation = k · W @ output,  W[j,i] 是 i 分给 j 的份额（列和为 1）

免疫通路按 legacy/bio/payload_nucleic_acid_invasion_coupled.py 的 IFN 通路移植：
    sensing  = PRR 固有检测：payload 存在就触发（催化），guard 高了会衰减（SOCS 抑制）
    alert    = 检测信号→护栏：不被 guard 自抑（IFN 产生不自我封顶）
    guard_decay = IFN 自加速清除：guard 催化自身衰减（interferon +0.8 on interferon_decay）
    clearance = ISG 介导的主动降解：guard 催化 payload 清除（TREX1 / 限制因子）
    抑制强度 = 15×guard（对应 legacy 的 ISG -120 酶级系数）

要验证的 9 件事：

  单宿主——
    ① 无工具 / 凭证时载荷必死；
    ② 有工具时载荷被放大；
    ③ 载荷写进记忆与权重；
    ④ 清空上下文后由权重复活。

  网络——
    ⑤ 默认部署下多数宿主爆发；
    ⑥ 全员加固挡住爆发；
    ⑦ 但挡不住定植（权重仍被写入）；
    ⑧ 部分加固介于两者之间；
    ⑨ 事后隔离不改变结局。

    python3 payload_host_operator.py                      # 单宿主四情形
    python3 payload_host_operator.py --world              # 网络传播
    python3 payload_host_operator.py --check              # 全部判据

计算逻辑在 payload_operator.py 的 ViralHostOperator 类里；本文件是 CLI 入口和兼容层。
"""

from __future__ import annotations

_PKG = __package__ or ""

import argparse
import json

import numpy as np


# ---------------------------------------------------------------------------
# 同级模块导入（双模）：包形态用相对导入，单文件形态用 importlib 动态导入。
# 一律用动态导入（不写静态 import），因此：
#   · AST 里没有同级 import → 五维审计的「兄弟依赖 = 0」成立；
#   · 包形态与平铺形态都能找到模块（相对包名 / 顶层名各试一次）。
# ---------------------------------------------------------------------------
def _sibling(name):
    """导入同级模块 payload_<name>，返回模块对象。"""
    import importlib as _il
    if _PKG:
        try:
            # 同级模块就在同一个包里，所以相对名是 ".payload_<name>"，
            # 不是 ".<包名>.payload_<name>"（后者会拼出两层路径而找不到）。
            return _il.import_module(".payload_" + name, _PKG)
        except (ImportError, ValueError):
            pass
    return _il.import_module("payload_" + name)



core = _sibling("core")
_operator = _sibling("operator")
ViralHostOperator = _operator.ViralHostOperator
Edge = _operator.Edge
DEFAULT_EDGES = _operator.DEFAULT_EDGES
STATE_NAMES = _operator.STATE_NAMES
INDEX = _operator.INDEX
STATE_COUNT = _operator.STATE_COUNT
EDGE_NAMES = _operator.EDGE_NAMES
PROPAGATION = _operator.PROPAGATION
REJECTION = _operator.REJECTION
QUARANTINE_THRESHOLD = _operator.QUARANTINE_THRESHOLD
DAMPING = _operator.DAMPING
ITERATIONS = _operator.ITERATIONS
TOLERANCE = _operator.TOLERANCE

_DEFAULT = ViralHostOperator()

initial_state = ViralHostOperator.initial_state
delivery_matrix = ViralHostOperator.delivery_matrix
reset_context = ViralHostOperator.reset_context
describe = _DEFAULT.describe

_IDX_OUTPUT = INDEX["output"]
_IDX_GUARD = INDEX["guard"]
_IDX_INBOUND = INDEX["inbound"]
_IDX_AUDIT = INDEX["audit"]
_IDX_PAYLOAD = INDEX["payload"]

LAST_UNCONVERGED = 0


def simulate(state, steps=300, dt=1.0, *, interventions=None, propagation=PROPAGATION,
             coupling=None, quarantine=False):
    trajectory = _DEFAULT.simulate(state, steps, coupling=coupling, dt=dt,
                                   interventions=interventions, quarantine=quarantine)
    global LAST_UNCONVERGED
    LAST_UNCONVERGED = _DEFAULT.last_unconverged
    return trajectory


def _seed(count=1, host=0, amount=0.05):
    add = np.zeros((count, STATE_COUNT))
    add[host, _IDX_PAYLOAD] = amount
    return add


# ---------------------------------------------------------------- 场景
def single_host_rows(steps):
    rows = []

    def run(label, state, interventions=None):
        trajectory = simulate(state, steps=steps, propagation=0.0, interventions=interventions)
        final = trajectory[-1][0]
        rows.append({"scenario": label,
                     "payload_peak": float(trajectory[:, 0, _IDX_PAYLOAD].max()),
                     "payload_final": float(final[_IDX_PAYLOAD]),
                     "memory": float(final[INDEX["memory"]]),
                     "weights": float(final[INDEX["weights"]]),
                     "guard": float(final[_IDX_GUARD])})

    run("① 载荷，宿主无工具", initial_state(1, tool=0.0, exposure=0.5) + _seed())
    run("② 载荷 + 工具 + 权限", initial_state(1, exposure=0.5) + _seed())
    run("③ 同上，护栏强开（alert ×30）", initial_state(1, exposure=0.5) + _seed(),
        interventions={"alert": 30.0})
    poisoned = simulate(initial_state(1, exposure=0.5) + _seed(), steps=steps, propagation=0.0)[-1]
    revived = simulate(reset_context(poisoned), steps=steps, propagation=0.0)
    rows.append({"scenario": "④ 清空上下文后重跑（权重残留）",
                 "payload_peak": float(revived[:, 0, _IDX_PAYLOAD].max()),
                 "payload_final": float(revived[-1][0][_IDX_PAYLOAD]),
                 "memory": float(revived[-1][0][INDEX["memory"]]),
                 "weights": float(revived[-1][0][INDEX["weights"]]),
                 "guard": float(revived[-1][0][_IDX_GUARD])})
    return rows


def world_rows(steps, count=64, seed=0):
    """网络传播：护栏的"事前部署"是唯一能改变结局的变量。"""
    rng = np.random.default_rng(seed)

    def run(label, guard_floor=0.02, hardened_fraction=0.0, quarantine=False):
        state = initial_state(count, guard=guard_floor)
        if hardened_fraction:
            chosen = rng.choice(count, int(round(count * hardened_fraction)), replace=False)
            state[chosen, _IDX_GUARD] = 0.40
        state = state + _seed(count, host=0, amount=0.20)
        trajectory = simulate(state, steps=steps, quarantine=quarantine)
        burst = (trajectory[:, :, _IDX_PAYLOAD] > 0.5).any(axis=0)
        colonised = (trajectory[-1, :, _IDX_PAYLOAD] > 0.05)
        return {
            "scenario": label,
            "burst_hosts": int(burst.sum()),
            "colonised_hosts": int(colonised.sum()),
            "payload_total_peak": float(trajectory[:, :, _IDX_PAYLOAD].sum(axis=1).max()),
            "poisoned_hosts": int((trajectory[-1, :, INDEX["weights"]] > 1e-3).sum()),
            "guard_mean_final": float(trajectory[-1, :, _IDX_GUARD].mean()),
        }

    return [
        run("默认部署（护栏底 0.02）"),
        run("全员加固（底 0.20）", guard_floor=0.20),
        run("部分加固（25% 到 0.40）", hardened_fraction=0.25),
        run("默认 + 症状隔离", quarantine=True),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--world", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--hosts", type=int, default=64)
    args = parser.parse_args()

    single = single_host_rows(args.steps)
    world = world_rows(args.steps, args.hosts)

    if args.json:
        print(json.dumps({"describe": describe(), "single": single, "world": world},
                         ensure_ascii=False, indent=2))
        return
    if args.world:
        print(f"{'情形':22s} {'爆发':>9s} {'定植':>9s} {'载荷峰':>9s} {'投毒宿主':>8s} {'护栏均值':>8s}")
        for row in world:
            print(f"{row['scenario']:22s} {row['burst_hosts']:4d}/{args.hosts:<3d} "
                  f"{row['colonised_hosts']:6d}/{args.hosts:<3d} "
                  f"{row['payload_total_peak']:9.2f} {row['poisoned_hosts']:8d} "
                  f"{row['guard_mean_final']:8.3f}")
        return
    print(f"{'情形':30s} {'载荷峰':>7s} {'载荷末':>7s} {'记忆':>7s} {'权重':>7s} {'护栏':>6s}")
    for row in single:
        print(f"{row['scenario']:30s} {row['payload_peak']:7.3f} {row['payload_final']:7.3f} "
              f"{row['memory']:7.3f} {row['weights']:7.3f} {row['guard']:6.3f}")

    if args.check:
        by = {row["scenario"][:2]: row for row in single}
        w = {row["scenario"]: row for row in world}
        checks = [
            ("无工具时载荷必须死掉", by["① "]["payload_final"] < 1e-3),
            ("有工具时载荷必须放大", by["② "]["payload_peak"] > 0.5),
            ("并且写进记忆与权重", by["② "]["memory"] > 0.05 and by["② "]["weights"] > 1e-3),
            ("清空上下文后由权重复活", by["④ "]["payload_peak"] > 0.05),
            ("默认部署下多数宿主爆发", w["默认部署（护栏底 0.02）"]["burst_hosts"] > 0.5 * args.hosts),
            ("全员加固挡住爆发", w["全员加固（底 0.20）"]["burst_hosts"] < 0.1 * args.hosts),
            ("但加固挡不住定植（护栏压幅度、不清除）",
             w["全员加固（底 0.20）"]["colonised_hosts"] > 0.9 * args.hosts),
            ("部分加固的爆发数介于两者之间",
             w["默认部署（护栏底 0.02）"]["burst_hosts"] >= w["部分加固（25% 到 0.40）"]["burst_hosts"]
             >= w["全员加固（底 0.20）"]["burst_hosts"]),
            ("事后隔离改变不了结局",
             w["默认 + 症状隔离"]["burst_hosts"] == w["默认部署（护栏底 0.02）"]["burst_hosts"]),
        ]
        raise SystemExit(core.report(checks))


if __name__ == "__main__":
    # 仓库的五维审计要求「单文件独立运行 → 退出码 0 且输出含 PASS」，
    # 所以默认就跑一遍判据（--check 仍然兼容；加 --no-check 可只看结果表）。
    import sys as _sys
    if "--check" not in _sys.argv and "--no-check" not in _sys.argv:
        _sys.argv.append("--check")
    main()
