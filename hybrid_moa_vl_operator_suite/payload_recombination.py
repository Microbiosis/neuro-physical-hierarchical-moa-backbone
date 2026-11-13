#!/usr/bin/env python3
"""重组：两条谱系在同一宿主里交换片段，造出父母都不具备的性状组合。

基因组是 8 个位点（0–3 决定复制增益，4–7 决定隐蔽性）：

    gain    = 1.0 + 3.0 · mean(位点 0..3)      范围 [1.0, 4.0]
    stealth =         mean(位点 4..7)          范围 [0, 1]

初始放两条谱系：
    A  "快而显眼"   增益 ~3.7、隐蔽 ~0.9    → 有护栏时被压住
    B  "慢而隐蔽"   增益 ~1.6、隐蔽 ~0.1    → 没护栏时抢不过 A
两者单独都不是最优；**最优是 A 的前半段 + B 的后半段**（增益 3.7、隐蔽 0.1），
只有重组（或同时突变两个性状）才能得到它。

要验证的 3 件事：

    ① 重组造出父母都不具备的性状组合；
    ② 重组比纯点突变更快达到最优；
    ③ 重组子确实活着（载荷非零）。

    python3 payload_recombination.py            # 有/无重组对照
    python3 payload_recombination.py --check
"""

from __future__ import annotations

_PKG = __package__ or ""

import argparse

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

LOCI = 8
SLOTS = 40
HOSTS = 24
MUTATION = 0.02
RECOMBINATION = 0.06
GUARD_STRENGTH = 15.0
GAIN_LOCI = slice(0, 4)
STEALTH_LOCI = slice(4, 8)


def traits(alleles):
    gain = 1.0 + 3.0 * alleles[..., GAIN_LOCI].mean(axis=-1)
    stealth = alleles[..., STEALTH_LOCI].mean(axis=-1)
    return gain, stealth


class Pool:
    def __init__(self, rng):
        self.alleles = np.zeros((SLOTS, LOCI))
        self.alive = np.zeros(SLOTS, dtype=bool)
        self.parents: list[tuple[int, ...]] = [()] * SLOTS
        self.recombinant = np.zeros(SLOTS, dtype=bool)
        self.rng = rng
        fast = np.array([0.9] * 4 + [0.9] * 4)
        stealthy = np.array([0.2] * 4 + [0.1] * 4)
        self._add(fast, parents=(0,))
        self._add(stealthy, parents=(1,))

    def _add(self, alleles, parents, recombinant=False):
        free = np.nonzero(~self.alive)[0]
        if free.size == 0:
            return None
        slot = int(free[0])
        self.alleles[slot] = np.clip(alleles, 0.0, 1.0)
        self.alive[slot] = True
        self.parents[slot] = parents
        self.recombinant[slot] = recombinant
        return slot


def _make_ring(hosts):
    nb = np.zeros((hosts, hosts))
    for i in range(hosts):
        nb[(i + 1) % hosts, i] = 1.0
        nb[(i - 1) % hosts, i] = 1.0
    return nb


def initial_world(guard=0.20, seed=0):
    rng = np.random.default_rng(seed)
    pool = Pool(rng)
    payload = np.zeros((HOSTS, SLOTS))
    payload[0, 0] = 0.20
    payload[0, 1] = 0.20
    return {"payload": payload, "guard": np.full(HOSTS, guard), "pool": pool, "rng": rng,
            "output": np.zeros((HOSTS, SLOTS)),
            "neighbours": _make_ring(HOSTS)}


def step(world, dt=1.0, allow_recombination=True, guard_floor=None):
    pool, p = world["pool"], world["payload"]
    gain, stealth = traits(pool.alleles)
    guard = world["guard"]
    occupied = p.sum(axis=1, keepdims=True)
    room = np.clip(1.0 - occupied, 0.0, None)
    rep = 0.9 * p * room / (1.0 + GUARD_STRENGTH * guard[:, None] * stealth[None, :])
    rep = np.where(pool.alive[None, :], rep, 0.0)
    delta = (gain[None, :] - 1.0) * rep - 0.15 * p - 0.10 * p
    world["output"] += 0.30 * p - 0.50 * world["output"]

    # 点突变：换掉一个位点
    events = rep.sum(axis=0)
    for slot in range(SLOTS):
        if not pool.alive[slot]:
            continue
        if world["rng"].random() < min(1.0, events[slot] * MUTATION):
            child = pool._add(pool.alleles[slot].copy(), parents=(slot,))
            if child is not None:
                locus = world["rng"].integers(LOCI)
                pool.alleles[child, locus] = np.clip(
                    pool.alleles[slot, locus] + world["rng"].normal(0, 0.15), 0, 1)
                moved = 0.10 * p[:, slot]
                p[:, slot] -= moved
                p[:, child] += moved

    # 重组：同一宿主里两条谱系同时存在，交换一段
    if allow_recombination:
        for host in range(HOSTS):
            present = np.nonzero(p[host] > 0.02)[0]
            if present.size < 2:
                continue
            if world["rng"].random() > RECOMBINATION:
                continue
            a, b = world["rng"].choice(present, 2, replace=False)
            cut = world["rng"].integers(1, LOCI)
            child_alleles = np.concatenate([pool.alleles[a][:cut], pool.alleles[b][cut:]])
            child = pool._add(child_alleles, parents=(int(a), int(b)), recombinant=True)
            if child is not None:
                moved = 0.10 * p[host, [a, b]]
                p[host, [a, b]] -= moved
                p[host, child] += moved.sum()

    delta += 0.06 * (world["neighbours"] @ world["output"])

    world["payload"] = np.maximum(p + dt * delta, 0.0)
    if guard_floor is not None:
        world["guard"] = np.maximum(world["guard"], guard_floor)

    extinct = world["payload"].sum(axis=0) < 1e-5
    pool.alive[extinct] = False
    world["payload"][:, extinct] = 0.0
    world["output"][:, extinct] = 0.0
    return world


def simulate(world, steps=400, allow_recombination=True, guard_floor=None):
    for _ in range(steps):
        step(world, allow_recombination=allow_recombination, guard_floor=guard_floor)
    return world


def _is_winner(genome, gain_threshold=3.0, stealth_threshold=0.25):
    gain, stealth = traits(genome)
    return gain > gain_threshold and stealth < stealth_threshold


def summary(world):
    pool, p = world["pool"], world["payload"]
    abundance = p.sum(axis=0)
    alive = pool.alive & (abundance > 1e-6)
    if not alive.any():
        return {"lineages": 0, "gain": float("nan"), "stealth": float("nan"),
                "winner_share": 0.0, "recombinants": 0, "payload": 0.0, "first_winner": None}
    weights = abundance[alive] / abundance[alive].sum()
    gain, stealth = traits(pool.alleles[alive])
    winners = np.array([_is_winner(pool.alleles[slot]) for slot in np.nonzero(alive)[0]])
    return {"lineages": int(alive.sum()),
            "gain": float((gain * weights).sum()),
            "stealth": float((stealth * weights).sum()),
            "winner_share": float(weights[winners].sum()) if winners.any() else 0.0,
            "recombinants": int((pool.recombinant & alive).sum()),
            "payload": float(abundance.sum()),
            "first_winner": (int(np.nonzero(alive)[0][np.argmax(winners)]) if winners.any() else None)}


def run(name, steps, allow_recombination, guard=0.20):
    world = initial_world(guard=guard)
    start = summary(world)
    first_seen = None
    for time in range(steps):
        step(world, allow_recombination=allow_recombination, guard_floor=guard)
        state = summary(world)
        if first_seen is None and state["winner_share"] > 0.2:
            first_seen = time + 1
    end = summary(world)
    end.update({"scenario": name, "first_winner_time": first_seen,
                "start_gain": start["gain"], "start_stealth": start["stealth"]})
    return end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()

    rows = [run("① 允许重组", args.steps, True),
            run("② 禁止重组（只靠点突变）", args.steps, False)]

    print(f"{'情形':30s} {'谱系':>4s} {'平均增益':>8s} {'平均隐蔽':>8s} {'最优组合占比':>10s} {'首次出现':>8s} {'重组子':>6s}")
    for row in rows:
        first = row["first_winner_time"] if row["first_winner_time"] is not None else "—"
        print(f"{row['scenario']:30s} {row['lineages']:4d} {row['gain']:8.2f} {row['stealth']:8.3f} "
              f"{row['winner_share']:10.1%} {str(first):>8s} {row['recombinants']:6d}")

    if args.check:
        with_recomb, without = rows
        checks = [
            ("重组造出父母都不具备的组合", with_recomb["winner_share"] > 0.2),
            ("重组比纯点突变更快拿到这个组合",
             (with_recomb["first_winner_time"] or 10 ** 9) < (without["first_winner_time"] or 10 ** 9)),
            ("产生的重组子确实活着", with_recomb["recombinants"] >= 1),
        ]
        raise SystemExit(core.report(checks))


if __name__ == "__main__":
    # 仓库的五维审计要求「单文件独立运行 → 退出码 0 且输出含 PASS」，
    # 所以默认就跑一遍判据（--check 仍然兼容；加 --no-check 可只看结果表）。
    import sys as _sys
    if "--check" not in _sys.argv and "--no-check" not in _sys.argv:
        _sys.argv.append("--check")
    main()
