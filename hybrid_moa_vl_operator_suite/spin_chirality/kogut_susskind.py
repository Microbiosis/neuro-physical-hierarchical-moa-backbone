# -*- coding: utf-8 -*-
"""1+1D Kogut–Susskind 规范理论的静态势：禁闭与弦张力。

做法（严格，不是拟合）
--------------------
把电荷 +1 放在 x=0、−1 放在 x=d（环上）。高斯定律

    E_x − E_{x−1} = ρ_x        （每个键 x 的电场减去前一个 = 该点的电荷）

在环上把 E 解出来：`E_x = c + (x 左侧累计电荷)`，`c` 是任意整数常数（它就是真空角 / θ 的
标签）。于是同一组电荷的**所有**物理态构成一个以 c 为指标的族，静态势就是

    V(d) = min_{c ∈ ℤ} (g²/2) Σ_x E_x²

对 d < N/2 这个最小值在 c = 0 取到，V(d) = (g²/2)·d：
**能量随间距线性增长 ⇒ 禁闭，弦张力 σ = g²/2**（格点单位，强耦合精确）。

这里不需要截断 E 的谱：c 只需枚举有限个（|c| ≤ d 就够），结果是精确的。
"""
from __future__ import annotations

import numpy as np


def _field_config(N: int, d: int, c: int) -> np.ndarray:
    """环上 +1（x=0）与 −1（x=d）两个电荷、真空常数 c 时的电场配置 E_x（长度 N）。"""
    rho = np.zeros(N)
    rho[0] += 1.0
    rho[d % N] -= 1.0
    E = np.empty(N)
    acc = 0.0
    for x in range(N):
        acc += rho[x]
        E[x] = c + acc
    return E


def static_potential(N: int, g: float, d: int, c_max: int | None = None):
    """相距 d 的静态电荷对的基态能量（对 c 取最小）及其对应的 c。"""
    if c_max is None:
        c_max = d + 2
    best, best_c = np.inf, 0
    for c in range(-c_max, c_max + 1):
        E = _field_config(N, d, c)
        v = 0.5 * g * g * float(np.sum(E * E))
        if v < best:
            best, best_c = v, c
    return float(best), int(best_c)


def check_confinement(N: int = 64, g: float = 1.0, d_max: int = 12) -> dict:
    """P33：禁闭——静态势 V(d) = (g²/2)·d，弦张力 σ = g²/2。

    判据：每个间距都取到 c=0 且 V(d) = (g²/2)d（残差 0），并报线性拟合的斜率。
    """
    sigma = 0.5 * g * g
    resid = 0.0
    for d in range(1, d_max + 1):
        v, c = static_potential(N, g, d)
        resid = max(resid, abs(v - sigma * d), float(abs(c)))
    out = {"V(d) = (g²/2)d（线性势）": resid}

    # 真空（无电荷）的能量：c=0 时严格为 0；c≠0 给出 θ 扇区的能量
    vac = [0.5 * g * g * float(np.sum(np.full(N, float(c)) ** 2)) for c in (0, 1, 2)]
    out["真空 (c=0) 能量为 0"] = abs(vac[0])
    out["θ 扇区 E(c) = (g²/2)·N·c²"] = abs(vac[1] - 0.5 * g * g * N) + abs(vac[2] - 2.0 * g * g * N)

    ds = np.arange(1, d_max + 1, dtype=float)
    vs = np.array([static_potential(N, g, int(d))[0] for d in ds])
    slope = float(np.polyfit(ds, vs, 1)[0])
    out["弦张力 σ_magnitude"] = slope
    out["σ vs g²/2_magnitude"] = sigma
    out["σ 与 g²/2 的偏差"] = abs(slope - sigma) / sigma
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check_confinement(), ensure_ascii=False, indent=2))
