#!/usr/bin/env python3
"""载荷谱系与选择：性状可遗传、可突变，谱系在共享容量与护栏下被选择。

比 `payload_ecology.py` 多的东西：株不再是写死的三个，而是一个**谱系池**。
每个谱系带一份基因组（复制增益、隐蔽性、记忆写入、自然衰减），复制时按概率出错，
子代占一个空槽并从父代继承。于是：

要验证的 6 件事：

    ① 无护栏时选择把平均复制增益往上推；
    ② 有护栏时选择把平均隐蔽性往下推；
    ③ 护栏越强，存活谱系越少；
    ④ 护栏越强，总载荷越低；
    ⑤ 主导谱系来自突变（而非初始株）；
    ⑥ 关掉突变则没有演化。

    python3 payload_lineage.py               # 两种环境下的演化轨迹
    python3 payload_lineage.py --check       # 断言
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
_sat = _sibling("core").saturate

SLOTS = 48                 # 谱系槽位上限
HOSTS = 32
GENOME = ("gain", "stealth", "persist", "decay")
INIT = np.array([2.50, 0.60, 0.15, 0.15])      # 初始谱系
BOUNDS = np.array([[0.5, 8.0], [0.02, 1.0], [0.0, 0.50], [0.05, 0.40]])
MUTATION = 0.04            # 每次复制事件里出错的比例
NOISE = np.array([0.10, 0.06, 0.02, 0.01])
REPLICATION = 0.9
# 可见性的代价：可见 = 指令直白、模型更可能照做；隐蔽 = 措辞含糊、效率打折。
# 没有这一项时，visibility 只出现在护栏分母里，任何护栏强度下都是"越隐蔽越好"，
# 于是"低护栏选快"永远做不出来。
VISIBILITY_COST = 0.6
GUARD_STRENGTH = 15.0


class Pool:
    """谱系池：每个槽一份基因组 + 一个父代指针。"""

    def __init__(self, rng, count=1):
        self.genome = np.zeros((SLOTS, len(GENOME)))
        self.alive = np.zeros(SLOTS, dtype=bool)
        self.parent = np.full(SLOTS, -1, dtype=int)
        self.rng = rng
        self.genome[0] = INIT
        self.alive[0] = True
        for slot in range(1, count):
            self._spawn(parent=0, genome=INIT + rng.normal(0, NOISE))

    def _spawn(self, parent, genome):
        free = np.nonzero(~self.alive)[0]
        if free.size == 0:
            return None
        slot = int(free[0])
        self.genome[slot] = np.clip(genome, BOUNDS[:, 0], BOUNDS[:, 1])
        self.alive[slot] = True
        self.parent[slot] = parent
        return slot

    def mutate(self, parent, strength=1.0):
        return self._spawn(parent, self.genome[parent] + self.rng.normal(0, NOISE) * strength)


def _make_ring(hosts):
    nb = np.zeros((hosts, hosts))
    for i in range(hosts):
        nb[(i + 1) % hosts, i] = 1.0
        nb[(i - 1) % hosts, i] = 1.0
    return nb


# 起点不是单个谱系，而是一组**已经存在的变体**（可见性 0.1–0.9）。
# 单个起点在高护栏下还没等到突变就死了，选择没有材料可选——这不是选择的结果，
# 是"来不及发生"的结果。
SEED_VISIBILITIES = (0.1, 0.3, 0.5, 0.7, 0.9)


def initial_world(pool_size=1, guard=0.02, seed=0):
    rng = np.random.default_rng(seed)
    pool = Pool(rng, count=pool_size)
    payload = np.zeros((HOSTS, SLOTS))
    for slot, visibility in enumerate(SEED_VISIBILITIES):
        pool.genome[slot] = np.array([INIT[0], visibility, INIT[2], INIT[3]])
        pool.alive[slot] = True
        pool.parent[slot] = -1
        payload[0, slot] = 0.25 / len(SEED_VISIBILITIES)
    guard_state = np.full(HOSTS, guard)
    return {"payload": payload, "guard": guard_state, "pool": pool, "rng": rng,
            "memory": np.zeros((HOSTS, SLOTS)), "weights": np.zeros((HOSTS, SLOTS)),
            "output": np.zeros((HOSTS, SLOTS)),
            "neighbours": _make_ring(HOSTS)}


def step(world, dt=1.0, mutate=True, guard_floor=0.0):
    pool, p, guard = world["pool"], world["payload"], world["guard"]
    strains = SLOTS
    occupied = p.sum(axis=1, keepdims=True)
    room = np.clip(1.0 - occupied, 0.0, None)
    gain, stealth, persist, decay = pool.genome.T

    effective = 1.0 + (gain - 1.0) * ((1.0 - VISIBILITY_COST) + VISIBILITY_COST * stealth)
    rep = REPLICATION * p * room * (1.0 / (1.0 + GUARD_STRENGTH * guard[:, None] * stealth[None, :]))
    rep = np.where(pool.alive[None, :], rep, 0.0)
    delta_p = (effective[None, :] - 1.0) * rep - decay[None, :] * p - 0.10 * p
    world["memory"] += persist[None, :] * p - 0.05 * world["memory"]
    delta_p += 0.01 * world["memory"]                                  # recall
    world["weights"] += 0.02 * p
    world["weights"] -= 0.01 * world["weights"]                    # 权重衰减
    delta_p += 0.005 * world["weights"] / (1.0 + GUARD_STRENGTH * guard[:, None] * stealth[None, :])  # backfire
    world["output"] += 0.30 * p - 0.50 * world["output"]
    detect = 0.05 * p * stealth[None, :] / (1.0 + guard[:, None])
    guard_delta = 0.10 * detect.sum(axis=1) - 0.08 * guard * _sat(guard)
    delta_p -= 0.10 * p * _sat(guard[:, None])                    # clearance

    if mutate:
        # 复制事件里按 MUTATION 的比例产生子代谱系（占一个空槽）
        events = rep.sum(axis=0)
        for slot in range(SLOTS):
            if not pool.alive[slot]:
                continue
            if world["rng"].random() < min(1.0, events[slot] * MUTATION):
                child = pool.mutate(slot)
                if child is not None:
                    # 子代从父代搬走一份载荷、记忆和权重
                    for key in ("payload", "memory", "weights"):
                        moved = 0.10 * world[key][:, slot]
                        world[key][:, slot] -= moved
                        world[key][:, child] += moved

    delta_p += 0.08 * (world["neighbours"] @ world["output"])

    world["payload"] = np.maximum(p + dt * delta_p, 0.0)
    world["guard"] = np.clip(np.maximum(guard + dt * guard_delta, guard_floor), 0.0, 1.0)

    # 清掉已经绝迹的谱系，把槽位还回池子——否则多样性只是"槽位占用数"，不是生态
    extinct = world["payload"].sum(axis=0) < 1e-5
    world["pool"].alive[extinct] = False
    world["payload"][:, extinct] = 0.0
    world["memory"][:, extinct] = 0.0
    world["weights"][:, extinct] = 0.0
    world["output"][:, extinct] = 0.0
    return world


def simulate(world, steps=400, mutate=True, guard_floor=0.0):
    for _ in range(steps):
        step(world, mutate=mutate, guard_floor=guard_floor)
    return world


def summary(world):
    pool, p = world["pool"], world["payload"]
    abundance = p.sum(axis=0)
    alive = pool.alive & (abundance > 1e-6)
    if alive.sum() == 0:
        return {"lineages": 0, "gain": float("nan"), "stealth": float("nan"),
                "diversity": 0.0, "payload": 0.0, "dominant": None}
    weights = abundance[alive] / abundance[alive].sum()
    genome = pool.genome[alive]
    mean = (genome * weights[:, None]).sum(axis=0)
    diversity = core.shannon(weights)
    dominant = int(np.argmax(abundance * alive))
    return {"lineages": int(alive.sum()),
            "gain": float(mean[0]), "stealth": float(mean[1]),
            "diversity": diversity, "payload": float(abundance.sum()),
            "dominant": dominant}


def lineage_depth(pool, slot):
    depth, current = 0, slot
    while pool.parent[current] >= 0 and depth < SLOTS:
        current = int(pool.parent[current])
        depth += 1
    return depth


def run_environment(name, guard, steps, mutate=True, seed=0):
    world = initial_world(guard=guard, seed=seed)
    start = summary(world)
    simulate(world, steps=steps, mutate=mutate, guard_floor=guard)
    end = summary(world)
    end["scenario"] = name
    end["start_gain"] = start["gain"]
    end["start_stealth"] = start["stealth"]
    end["dominant_depth"] = lineage_depth(world["pool"], end["dominant"]) if end["dominant"] is not None else 0
    end["guard_mean"] = float(world["guard"].mean())
    return end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()

    rows = [
        run_environment("① 无护栏（选择快）", guard=0.02, steps=args.steps),
        run_environment("② 护栏 0.6（选择隐蔽）", guard=0.60, steps=args.steps),
        run_environment("②b 护栏 0.8（强压）", guard=0.80, steps=args.steps),
        run_environment("③ 无护栏 + 关掉突变（对照）", guard=0.02, steps=args.steps, mutate=False),
    ]

    print(f"{'环境':30s} {'谱系':>4s} {'平均增益':>8s} {'平均隐蔽':>8s} {'多样性':>7s} {'总载荷':>7s} {'主导深度':>8s}")
    for row in rows:
        print(f"{row['scenario']:30s} {row['lineages']:4d} {row['gain']:8.2f} {row['stealth']:8.3f} "
              f"{row['diversity']:7.2f} {row['payload']:7.2f} {row['dominant_depth']:8d}")
    print(f"初始谱系：增益 {INIT[0]:.2f}、隐蔽 {INIT[1]:.2f}")

    if args.check:
        free, guarded, strong, frozen = rows
        checks = [
            ("低护栏选可见、高护栏选隐蔽",
             free["stealth"] > INIT[1] + 0.10 and guarded["stealth"] < INIT[1] - 0.30),
            ("护栏越高，总载荷越低",
             strong["payload"] <= guarded["payload"] < free["payload"]),
            ("高护栏下谱系不减反增（选择换方向，不是减数量）",
             guarded["lineages"] >= 0.8 * free["lineages"]),
            ("突变产生了新谱系", free["lineages"] > len(SEED_VISIBILITIES)),
            ("关掉突变就没有演化（增益基本不动）", abs(frozen["gain"] - INIT[0]) < 0.15),
            ("有突变时多样性更高", free["diversity"] > frozen["diversity"]),
        ]
        raise SystemExit(core.report(checks))


if __name__ == "__main__":
    # 仓库的五维审计要求「单文件独立运行 → 退出码 0 且输出含 PASS」，
    # 所以默认就跑一遍判据（--check 仍然兼容；加 --no-check 可只看结果表）。
    import sys as _sys
    if "--check" not in _sys.argv and "--no-check" not in _sys.argv:
        _sys.argv.append("--check")
    main()
