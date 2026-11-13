#!/usr/bin/env python3
"""投递语义 → 阈值：把"真实通道"翻译成两个可测的量。

上一层的结论是"阈值方向由投递语义决定"。这一层把投递语义写成可测参数：

    fan-out      一条消息平均发给多少人（1 = 私信，8 = 群聊，N = 广播）
    per-recipient  每个接收者拿到全量（broadcast）还是 1/fan-out（share）
    success      单次投递让接收方"开始复制"的概率（这是可测的：注入成功率）

于是问题变成一句可回答的话：
    **一个 fan-out 为 k 的通道，注入成功率要多少，载荷才起得来？**

要验证的 4 件事：

    ① 全量投递比份额投递更容易起；
    ② 同口径下扇出越大越容易起；
    ③ 广播需要最低成功率；
    ④ 信息流比群聊更容易起。

    python3 viral_channels.py            # 通道 × 阈值
    python3 viral_channels.py --check
"""

from __future__ import annotations

import argparse

import numpy as np

import payload_core as core
import viral_spatial as S

HOSTS = S.HOSTS          # 与空间层同一套宿主数，直接复用它的模拟器


def channel_graph(kind, hosts=HOSTS, fanout=4, seed=0):
    """通道翻译成图：私信=环形 1 邻、群聊=环形 fanout/2 邻、信息流=Zipf、广播=全连。"""
    if kind == "dm":
        return core.ring(hosts, 1)
    if kind in ("group", "group_full"):
        return core.ring(hosts, max(1, fanout // 2))
    if kind == "feed":
        return core.feed(hosts, seed=seed)
    if kind == "broadcast":
        return core.complete(hosts)
    raise ValueError(kind)


def critical_success(adjacency, mode, low=0.002, high=0.30, steps=160, target=0.2):
    """二分：单次投递的成功率要多少，才能让 target 比例的宿主被点亮。"""
    saved_mode, saved_transmission = S.MODE, S.TRANSMISSION
    try:
        def outbreak(rate):
            S.MODE = mode
            S.TRANSMISSION = rate
            world = S.simulate(adjacency, steps=steps)
            return (world["payload"].sum(axis=1) > 0.02).sum() / adjacency.shape[0]
        return core.bisect(outbreak, low, high, target)
    finally:
        S.MODE, S.TRANSMISSION = saved_mode, saved_transmission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sensitivity", action="store_true", help="宿主内清除强度对阈值的影响")
    args = parser.parse_args()

    rows = []
    for kind, fanout, mode in (
            ("dm", 1, "broadcast"),        # 私信：一个人，拿全量
            ("group", 6, "share"),         # 群聊：六个人分这条消息的注意力
            ("group_full", 6, "broadcast"),  # 对照：同样六个人，但每人拿到全文
            ("feed", 6, "broadcast"),      # 信息流：粉丝各看全文
            ("broadcast", HOSTS - 1, "broadcast")):
        adjacency = channel_graph(kind, fanout=fanout)
        degree = float(adjacency.sum(axis=1).mean())
        rows.append({"channel": kind, "mean_fanout": degree, "mode": mode,
                     "critical_success": critical_success(adjacency, mode)})

    print(f"{'通道':10s} {'平均扇出':>8s} {'口径':>10s} {'临界单次成功率':>14s}")
    for row in rows:
        print(f"{row['channel']:10s} {row['mean_fanout']:8.1f} {row['mode']:>10s} "
              f"{row['critical_success']:14.3f}")

    if args.sensitivity:
        adjacency = channel_graph("group_full", fanout=6)
        print("同样的通道（群播、扇出 6），只改宿主内的清除强度：")
        base = S.R.GUARD_STRENGTH
        for guard in (6.0, 12.0, 24.0):
            S.R.GUARD_STRENGTH = guard
            print(f"  λ={guard:4.0f}  临界成功率 {critical_success(adjacency, 'broadcast'):.4f}")
        S.R.GUARD_STRENGTH = base
        return

    if args.check:
        by = {row["channel"]: row for row in rows}
        checks = [
            ("同样的扇出，每人拿到全量比分成份额更容易起",
             by["group_full"]["critical_success"] < by["group"]["critical_success"]),
            ("同样口径下，扇出越大越容易起（群播 < 私信）",
             by["group_full"]["critical_success"] < by["dm"]["critical_success"]),
            ("广播需要的最低", by["broadcast"]["critical_success"]
             <= min(row["critical_success"] for row in rows)),
            ("信息流（少数人看全文）比群聊（大家分份额）容易起",
             by["feed"]["critical_success"] <= by["group"]["critical_success"]),
        ]
        raise SystemExit(core.report(checks))


if __name__ == "__main__":
    main()
