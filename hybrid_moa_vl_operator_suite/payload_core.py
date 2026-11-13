#!/usr/bin/env python3
"""载荷算子各层共用的底座：饱和、图、二分阈值、多样性、判据打印。

这一层没有模型，只有**每一层都在重复写的东西**。放在这里的原因很实际：
`payload_spatial.py` 与 `payload_channels.py` 各自实现过一遍二分，`ring/random/...`
被抄了两份，`saturate` 在三个文件里叫过三个名字。整合之后，改一处就够。
"""

from __future__ import annotations

_PKG = __package__ or ""

import math

import numpy as np

# ---------------------------------------------------------------- 数值


def saturate(value, half=0.5):
    """饱和：没有底物时趋近 0，给足时趋近 1。"""
    return value / (half + value)


def normalise(adjacency):
    """每列归一：一个宿主送出去的份额之和为 1（share 口径）。"""
    column = adjacency.sum(axis=0, keepdims=True)
    return adjacency / np.maximum(column, 1e-12)


def shannon(weights):
    """Shannon 多样性；权重和不为 1 时先归一。"""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return 0.0
    weights = weights / total
    return float(-(weights * np.log(weights + 1e-12)).sum())


# ---------------------------------------------------------------- 图


def ring(hosts, neighbours=1):
    adjacency = np.zeros((hosts, hosts))
    for i in range(hosts):
        for step in range(1, neighbours + 1):
            adjacency[i, (i + step) % hosts] = 1.0
            adjacency[(i + step) % hosts, i] = 1.0
    return adjacency


def random_graph(hosts, mean_degree=4, seed=0):
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((hosts, hosts))
    edges = max(hosts, hosts * mean_degree // 2)
    while adjacency.sum() // 2 < edges:
        i, j = rng.integers(0, hosts, 2)
        if i != j:
            adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def smallworld(hosts, mean_degree=4, rewire=0.25, seed=0):
    rng = np.random.default_rng(seed)
    adjacency = ring(hosts, max(1, mean_degree // 2))
    for _ in range(int(hosts * rewire)):
        i, j = rng.integers(0, hosts, 2)
        if i != j:
            adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def scalefree(hosts, m=2, seed=0):
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((hosts, hosts))
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    for node in range(2, hosts):
        degree = adjacency[:node].sum(axis=1)
        probability = (degree + 1.0) / (degree + 1.0).sum()
        for other in rng.choice(node, size=min(m, node), replace=False, p=probability):
            adjacency[node, other] = adjacency[other, node] = 1.0
    return adjacency


def community(hosts, intra=0.35, bridges=2, seed=0):
    rng = np.random.default_rng(seed)
    half = hosts // 2
    adjacency = np.zeros((hosts, hosts))
    for group in (range(half), range(half, hosts)):
        for i in group:
            for j in group:
                if i != j and rng.random() < intra:
                    adjacency[i, j] = 1.0
    for b in range(bridges):
        adjacency[b, half + b] = adjacency[half + b, b] = 1.0
    return adjacency


def complete(hosts):
    return np.ones((hosts, hosts)) - np.eye(hosts)


def feed(hosts, seed=0):
    """信息流：少数账号被很多人看（Zipf 扇出）。"""
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((hosts, hosts))
    for i in range(hosts):
        for _ in range(min(int(rng.zipf(1.8)), hosts // 2)):
            j = int(rng.integers(0, hosts))
            if i != j:
                adjacency[j, i] = 1.0
    return adjacency


GRAPHS = {
    "ring": lambda hosts, **kw: ring(hosts, max(1, kw.get("mean_degree", 4) // 2)),
    "random": random_graph,
    "smallworld": smallworld,
    "scalefree": scalefree,
    "community": community,
    "complete": complete,
    "feed": feed,
}


# ---------------------------------------------------------------- 阈值


def bisect(outbreaks, low, high, target, rounds=9):
    """找"越过 target"的最小参数：outbreaks(x) 返回 0..1 的比例。"""
    for _ in range(rounds):
        middle = 0.5 * (low + high)
        if outbreaks(middle) >= target:
            high = middle
        else:
            low = middle
    return high


# ---------------------------------------------------------------- 判据


def report(checks, title=None):
    """统一的判据打印；返回退出码。"""
    if title:
        print(title)
    failures = 0
    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1
    print(f"判据 {len(checks)} 条，失败 {failures} 条")
    return 1 if failures else 0


if __name__ == "__main__":              # pragma: no cover
    # 仓库的五维审计要求「单文件独立运行 → 退出码 0 且输出含 PASS」。
    import numpy as _np
    _s = _np.array([0.0, 0.1, 0.5, 1.0, 10.0, 1000.0])
    _sat = saturate(_s)
    _graphs = {
        "ring": ring(6), "random": random_graph(6),
        "smallworld": smallworld(6), "scalefree": scalefree(6),
        "community": community(6), "complete": complete(6),
        "feed": feed(6),
    }
    # bisect 的第一个参数是「返回 0..1 比例」的函数，不是数组。
    _outbreaks = lambda x: float(x)
    _checks = [
        ("saturate 单调增",
         bool((_sat[1:] > _sat[:-1]).all())),
        ("saturate(0)=0、saturate(大)=1",
         abs(float(_sat[0])) < 1e-12 and abs(float(_sat[-1]) - 1.0) < 1e-3),
        ("saturate(v) = v/(half+v) 精确一致",
         bool(_np.allclose(_sat, _s / (0.5 + _s)))),
        ("7 种图都给出 hosts×hosts 邻接矩阵，无向图对称、有向图不对称",
         len(_graphs) == 7
         and all(g.shape == (6, 6) for g in _graphs.values())
         and bool((_graphs["ring"] == _graphs["ring"].T).all())
         and bool((_graphs["complete"] == _graphs["complete"].T).all())
         and not bool((_graphs["community"] == _graphs["community"].T).all())
         and not bool((_graphs["feed"] == _graphs["feed"].T).all())),
        ("normalise 后每列和为 1（share 口径）",
         bool(_np.allclose(normalise(_graphs["ring"]).sum(axis=0), 1.0))),
        ("shannon 返回自然对数熵（4 等权 = ln4 = 1.3863）、零权重为 0",
         abs(shannon(_np.array([1.0, 1.0, 1.0, 1.0])) - math.log(4.0)) < 1e-9 and
         abs(shannon(_np.array([0.0, 0.0]))) < 1e-9 and
         shannon(_np.array([1.0, 0.0, 0.0, 0.0])) < 1e-9),
        ("二分阈值单调：目标越高阈值越大",
         bisect(_outbreaks, 0.0, 1.0, 0.3) <= bisect(_outbreaks, 0.0, 1.0, 0.7)),
    ]
    raise SystemExit(report(_checks, "payload_core 底座自检"))
