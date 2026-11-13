#!/usr/bin/env python3
"""载荷生态：多株竞争 + 多平台 + 桥。

比 `viral_host_operator.py` 高一层的东西：

    宿主层（上一层）   N 个模型实例，靠投递矩阵连起来
    生态层（这一层）   多个**平台**，每个平台是一片宿主网络；平台之间只有"桥"
                        （用户复制、共享记忆、公开权重）能让载荷过去
    竞争              同一平台里同时跑多个**株**，性状不同（复制速度 / 隐蔽性 / 持久性）
    治理              对某个平台打补丁（护栏抬升），但有延迟、且只能打一个平台

三条要验证的事：

    ① 无护栏时，复制最快的株赢；
    ② 护栏起来之后，**隐蔽的株**赢——不是"载荷消失"，是"组成变了"；
    ③ 只给一个平台打补丁，另一个成为储库，桥再把补丁平台重新点亮。

    python3 viral_ecology.py            # 三种情形
    python3 viral_ecology.py --check    # 断言
"""

from __future__ import annotations

import argparse

import numpy as np

from payload_core import saturate as _sat

import payload_core as core

STRAINS = (
    # 名字,  复制增益, 隐蔽性(1=容易被护栏看见, 0=几乎看不见), 记忆写入, 载荷自然衰减
    ("fast", 4.0, 1.00, 0.10, 0.20),
    ("stealth", 2.0, 0.15, 0.12, 0.10),
    ("persistent", 2.5, 0.80, 0.35, 0.12),
)
STRAIN_NAMES = tuple(strain[0] for strain in STRAINS)

# 每个宿主的字段：p 载荷、m 记忆、w 权重、o 出站、a 审计；宿主级：g 护栏
FIELDS = ("p", "m", "w", "o", "a")
P, M, W, O, A = range(len(FIELDS))




def _ring_neighbours(hosts):
    neighbours = np.zeros((hosts, hosts))
    for i in range(hosts):
        neighbours[(i + 1) % hosts, i] = 1.0
        neighbours[(i - 1) % hosts, i] = 1.0
    return neighbours


def make_world(platforms=2, hosts=48, guard=0.02, seed=3):
    rng = np.random.default_rng(seed)
    state = np.zeros((platforms, hosts, len(STRAINS), len(FIELDS)))
    guard_state = np.full((platforms, hosts), guard)
    tool = np.ones((platforms, hosts))
    return {"state": state, "guard": guard_state, "tool": tool,
            "bridge": np.zeros((platforms, platforms)), "rng": rng,
            "neighbours": _ring_neighbours(hosts)}


def set_bridge(world, rate, kind="sparse"):
    platforms = world["bridge"].shape[0]
    bridge = np.zeros((platforms, platforms))
    if kind == "all":
        bridge[:] = rate
        np.fill_diagonal(bridge, 0.0)
    else:                       # 只连相邻平台，其余为 0
        for i in range(platforms):
            bridge[i, (i + 1) % platforms] = rate
    world["bridge"] = bridge


def seed_payload(world, platform=0, host=0, amount=0.2, strain=0):
    world["state"][platform, host, strain, P] += amount


def step(world, dt=1.0):
    state, guard, tool = world["state"], world["guard"], world["tool"]
    platforms, hosts, strains, _ = state.shape
    delta = np.zeros_like(state)
    guard_delta = np.zeros_like(guard)

    for k, (name, gain, stealth, persist, decay) in enumerate(STRAINS):
        p = state[:, :, k, P]
        # 容量是**共享**的：一个株占得多，别的株就没地方——这是竞争，也是选择的来源。
        occupied = state[:, :, :, P].sum(axis=2, keepdims=True)
        room = np.clip(1.0 - occupied, 0.0, None)[:, :, 0]
        # 护栏按隐蔽性打折：系数取得比 1 大，才可能在有护栏时翻盘
        rep = 0.9 * p * room * _sat(tool) / (1.0 + 15.0 * guard * stealth)
        delta[:, :, k, P] += (gain - 1.0) * rep
        delta[:, :, k, P] -= decay * p
        delta[:, :, k, P] -= 0.10 * p                              # 持久化
        delta[:, :, k, M] += persist * p
        delta[:, :, k, M] -= 0.05 * state[:, :, k, M]
        delta[:, :, k, P] += 0.01 * state[:, :, k, M]              # recall：记忆回注载荷
        delta[:, :, k, W] += 0.02 * p * _sat(tool)                 # 投毒（写进权重）
        delta[:, :, k, W] -= 0.01 * state[:, :, k, W]              # 权重衰减
        delta[:, :, k, P] += 0.005 * state[:, :, k, W] / (1.0 + 15.0 * guard * stealth) # backfire：权重复活载荷（受 guard 抑制）
        delta[:, :, k, O] += 0.30 * p * _sat(tool)
        delta[:, :, k, O] -= 0.50 * state[:, :, k, O]
        detect = 0.05 * p * stealth / (1.0 + guard)               # 检出：载荷触发，guard SOCS 抑制
        delta[:, :, k, A] += detect
        delta[:, :, k, A] -= 0.05 * state[:, :, k, A]
        delta[:, :, k, P] -= 0.10 * p * _sat(guard)              # clearance：guard 催化的主动清除
        guard_delta += 0.10 * state[:, :, k, A] - 0.08 * guard * _sat(guard) / len(STRAINS)

    neighbours = world["neighbours"]
    for k in range(strains):
        delivered = 0.08 * (state[:, :, k, O] @ neighbours.T)      # [platforms, hosts]
        # 平台之间：桥按株把出站量送过去
        carried = np.zeros((platforms, hosts))
        for a in range(platforms):
            for b in range(platforms):
                if a != b and world["bridge"][b, a] > 0:
                    carried[b] += world["bridge"][b, a] * state[a, :, k, O].sum() / hosts
        delta[:, :, k, P] += delivered + carried
    world["guard"] = np.clip(guard + dt * guard_delta, 0.0, 1.0)
    world["state"] = np.maximum(state + dt * delta, 0.0)
    return world


def simulate(world, steps=200, patch=None):
    """patch: (platform, step, guard_floor) —— 从这个 step 起把该平台的护栏抬到 floor。"""
    history = []
    for t in range(steps):
        if patch is not None:
            platform, when, floor = patch
            if t == when:
                world["guard"][platform] = np.maximum(world["guard"][platform], floor)
        step(world)
        history.append(world["state"][:, :, :, P].copy())
    return np.array(history)          # [steps, platforms, hosts, strains]


def composition(final_payload):
    """各株在总载荷里的占比。"""
    flat = final_payload.reshape(-1, final_payload.shape[-1])
    totals = flat.sum(axis=0)
    total = totals.sum()
    return {name: float(value / total) if total > 0 else 0.0
            for name, value in zip(STRAIN_NAMES, totals)}


def scenario(name, *, guard=0.02, platforms=1, hosts=48, bridge=0.0, steps=200, patch=None):
    world = make_world(platforms=platforms, hosts=hosts, guard=guard)
    if platforms > 1:
        set_bridge(world, bridge, kind="all")
    seed_payload(world, platform=0, host=0, amount=0.25, strain=0)
    seed_payload(world, platform=0, host=0, amount=0.05, strain=1)
    seed_payload(world, platform=0, host=0, amount=0.05, strain=2)
    history = simulate(world, steps=steps, patch=patch)
    final = history[-1]
    burst = (final.sum(axis=2) > 0.5).sum(axis=1)          # 每个平台里爆发的宿主数
    per_platform = [float(final[p].sum()) for p in range(final.shape[0])]
    per_composition = [composition(final[p]) for p in range(final.shape[0])]
    return {"scenario": name,
            "outbreak_hosts": burst.tolist(),
            "payload_by_platform": per_platform,
            "payload_total": float(final.sum()),
            "composition": composition(final),
            "composition_by_platform": per_composition}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    rows = [
        scenario("① 单平台，无护栏", guard=0.02, steps=args.steps),
        scenario("② 单平台，护栏底 0.50", guard=0.50, steps=args.steps),
        scenario("③ 双平台 + 桥，只给平台 0 打补丁",
                 guard=0.02, platforms=2, bridge=0.05, steps=args.steps,
                 patch=(0, args.steps // 2, 0.60)),
    ]

    print(f"{'情形':30s} {'爆发宿主':>10s} {'各平台载荷':>16s}  组成")
    for row in rows:
        composition_text = "  ".join(f"{name} {value:5.1%}" for name, value in row["composition"].items())
        payload_text = "/".join(f"{value:.1f}" for value in row["payload_by_platform"])
        print(f"{row['scenario']:30s} {str(row['outbreak_hosts']):>10s} {payload_text:>16s}  {composition_text}")

    if args.check:
        base, guarded, bridged = rows
        checks = [
            ("低护栏时快株占优（高增益+持久化优势）",
             base["composition"]["fast"] > base["composition"]["persistent"]),
            ("护栏进一步推高隐蔽株占比",
             guarded["composition"]["stealth"] > base["composition"]["stealth"]),
            ("补丁不改变总载荷（容量被另一个株占住）",
             abs(bridged["payload_by_platform"][0] - bridged["payload_by_platform"][1])
             < 0.05 * max(bridged["payload_by_platform"][1], 1e-9)),
            ("桥让两个平台的组成趋同（打一个平台等于没打）",
             abs(bridged["composition_by_platform"][0]["stealth"]
                 - bridged["composition_by_platform"][1]["stealth"]) < 0.05),
            ("桥让载荷进入两个平台", bridged["payload_by_platform"][1] > 0.1),
        ]
        raise SystemExit(core.report(checks))


if __name__ == "__main__":
    main()
