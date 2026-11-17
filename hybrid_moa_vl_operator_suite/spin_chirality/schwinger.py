# -*- coding: utf-8 -*-
"""1+1D 规范场反作用（半经典）：等离子频率与电荷屏蔽，都等于 e/√π。

做法
----
Schwinger 模型里，无质量费米子的一圈诱导电流是**精确**的：

    j¹ = −(e/π) A₁            （均匀 A₁，长度归一）

把它喂回麦克斯韦方程（均匀场、均匀电流）：

    E = −∂_t A₁ ,        ∂_t E = −j¹

得

    ∂_t² A₁ + (e²/π) A₁ = 0     ⇒   ω_p = e/√π        光子获得质量 m_γ = e/√π

静态情形：`∂_x E = ρ − (e²/π)A₁` ⇒ `(∂_x² − m_γ²) A₀ = −ρ`，点电荷的场按 `e^{−m_γ|x|}` 衰减。

所以这一层给出**两个可测数字**（ω_p 与屏蔽长度 1/m_γ），都由同一个 e/√π 决定。
有质量时诱导电流变成非局域核（含手征凝聚的贡献），本模块只做无质量这一支——那是能精确写的。
"""
from __future__ import annotations

import numpy as np

_XG, _WG = np.polynomial.legendre.leggauss(400)
_XX, _WW = 0.5 * (_XG + 1.0), 0.5 * _WG


def plasma_frequency(e: float = 1.0) -> float:
    """ω_p = e/√π（Schwinger 光子质量 m_γ 的同一个数）。"""
    return e / np.sqrt(np.pi)


def simulate(T: float = 10.0, dt: float = 1e-3, e: float = 1.0,
             a0: float = 0.3, v0: float = 0.0):
    """半经典反作用的时间演化：返回 (t, A₁, E)。

        dA₁/dt = −E,      dE/dt = −j¹ = (e²/π) A₁

    用 RK4 积分（这是一个线性谐振子，RK4 的误差只体现在相位上，量级 dt⁴）。
    """
    n = int(round(T / dt))
    k = e * e / np.pi
    a, v = float(a0), -float(v0)          # v ≡ dA₁/dt = −E
    ts = np.empty(n + 1)
    As = np.empty(n + 1)
    Es = np.empty(n + 1)
    ts[0], As[0], Es[0] = 0.0, a, -v
    for i in range(n):
        def f(a_, v_):
            return v_, -k * a_
        k1a, k1v = f(a, v)
        k2a, k2v = f(a + 0.5 * dt * k1a, v + 0.5 * dt * k1v)
        k3a, k3v = f(a + 0.5 * dt * k2a, v + 0.5 * dt * k2v)
        k4a, k4v = f(a + dt * k3a, v + dt * k3v)
        a += dt * (k1a + 2 * k2a + 2 * k3a + k4a) / 6.0
        v += dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        ts[i + 1], As[i + 1], Es[i + 1] = (i + 1) * dt, a, -v
    return ts, As, Es


def measure_frequency(T: float = 10.0, dt: float = 1e-3, e: float = 1.0) -> float:
    """从 A₁(t) 的零点间隔测频率。"""
    ts, As, _ = simulate(T, dt, e, a0=0.3)
    zc = ts[1:-1][(As[1:-1] * As[2:] < 0)]
    if len(zc) < 3:
        return float("nan")
    periods = np.diff(zc) * 2.0           # 相邻零点相隔半个周期
    return float(2.0 * np.pi / np.mean(periods))


def screening_length(e: float = 1.0, L: float = 40.0, h: float = 0.02,
                     x1: float = 6.0, x2: float = 10.0) -> float:
    """点电荷的静态场衰减长度：解 (∂_x² − m_γ²)A₀ = −ρ，ρ = δ(x)。

    用两个远处的点估衰减（`ln|A(x1)/A(x2)|/(x2−x1)`），避开边界处 Dirichlet 条件的污染。
    盒子必须远大于 1/m_γ；L=12 时边界会把曲线压弯，测出的 ξ 会偏小 40%（踩过）。
    """
    n = int(round(L / h))
    x = (np.arange(n) - n // 2) * h
    m2 = e * e / np.pi
    main = np.full(n, -2.0 / h ** 2 - m2)
    off = np.full(n - 1, 1.0 / h ** 2)
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    A[0, 0] = A[-1, -1] = 1.0
    rho = np.zeros(n)
    rho[n // 2] = 1.0 / h                              # δ 的格点化
    sol = np.linalg.solve(A, -rho)
    i1 = int(round((x1 + L / 2) / h))
    i2 = int(round((x2 + L / 2) / h))
    m = float(np.log(abs(sol[i1]) / abs(sol[i2])) / (x2 - x1))
    return 1.0 / m


def check_backreaction() -> dict:
    """P31：规范场反作用——等离子频率与屏蔽长度都等于 e/√π。"""
    e = 1.0
    wp = measure_frequency(20.0, 1e-3, e)
    out = {"等离子频率 vs e/√π": abs(wp - plasma_frequency(e)) / plasma_frequency(e)}

    # 能量守恒：H = E²/2 + (e²/2π) A₁²
    ts, As, Es = simulate(20.0, 1e-3, e, a0=0.3)
    H = 0.5 * Es ** 2 + 0.5 * (e * e / np.pi) * As ** 2
    out["能量守恒"] = float(np.abs(H - H[0]).max() / abs(H[0]))

    # 屏蔽长度 = 1/m_γ = √π/e
    xi = screening_length(e)
    out["屏蔽长度 vs √π/e"] = abs(xi - np.sqrt(np.pi) / e) / (np.sqrt(np.pi) / e)
    out["屏蔽长度_magnitude"] = xi
    return out


# ------------------------------------------------------------------------------
# 有质量的反作用：一圈真空极化 Π(q²) 与色散方程 ω² = e²Π(ω²)
# ------------------------------------------------------------------------------
def vacuum_polarization(q2: float, m: float) -> float:
    """1+1D 单圈真空极化（一个 Dirac 费米子）的实部：

        Π(q²) = (1/π)[ 1 + ∫₀¹ dx · x(1−x)q² / (m² − x(1−x)q²) ]

    三个解析极限，正好是这一层要验的东西：
      · Π(0) = 1/π —— 与 m **无关**（反常的不可重整化）
      · Π(∞) = 0
      · q² → 4m²⁻ 时对数发散 ⇒ 对产生阈值
    本函数只在阈下（q² < 4m²）用，那里分母恒正、无奇点。
    """
    if q2 >= 4.0 * m * m - 1e-12:
        raise ValueError("本函数只用于阈下 q² < 4m²；阈上见 imag_polarization")
    x, w = _XX, _WW
    den = m * m - x * (1.0 - x) * q2
    return float((1.0 + np.sum(w * x * (1.0 - x) * q2 / den)) / np.pi)


def imag_polarization(q2: float, m: float) -> float:
    """阈上 q² > 4m² 的虚部（解析）：Im Π = m² / (q² √(1 − 4m²/q²))；阈下恒为 0。"""
    if q2 <= 4.0 * m * m:
        return 0.0
    return float(m * m / (q2 * np.sqrt(1.0 - 4.0 * m * m / q2)))


def photon_mass_massive(m: float, e: float = 1.0, n_grid: int = 400) -> float:
    """有质量时的光子质量：解色散方程 `ω² = e² Π(ω²)`（取阈下最低的那个根）。

    m 太小（≲0.4）时阈上 Π 的对数发散把根吞掉——没有稳定解，返回 nan，
    对应的物理是「光子质量 e/√π 落在对产生连续谱里」。
    """
    hi = 4.0 * m * m * (1.0 - 1e-6)
    ws = np.linspace(1e-12, hi, n_grid)
    vals = np.array([w - e * e * vacuum_polarization(w, m) for w in ws])
    idx = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(idx) == 0:
        return float("nan")
    i0 = idx[0]
    lo_, hi_ = ws[i0], ws[i0 + 1]
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_)
        if mid - e * e * vacuum_polarization(mid, m) > 0:
            hi_ = mid
        else:
            lo_ = mid
    return float(np.sqrt(0.5 * (lo_ + hi_)))


def check_massive() -> dict:
    """P32：有质量的反作用——Π(0) 与 m 无关（反常不可重整化）、阈上有虚部、ω_p 趋于 e/√π。"""
    out = {}
    # (a) Π(0) = 1/π，任意质量（这是「反常与质量无关」的数值化身）
    out["Π(0) = 1/π（与 m 无关）"] = max(
        abs(vacuum_polarization(1e-12, m) - 1.0 / np.pi) for m in (0.1, 0.5, 2.0))
    # (b) 阈值：阈下虚部严格为零；阈下端点处 Π 对数发散
    m = 0.7
    out["Im Π = 0（阈下）"] = max(abs(imag_polarization(q, m)) for q in (0.1, 1.0, 1.9))
    out["阈下端点 Π 发散（> 10/π）"] = max(
        0.0, 10.0 / np.pi - vacuum_polarization(0.999 * 4.0 * m * m, m))
    # (c) 有质量的光子质量：m 越大越靠近 e/√π（从上往下）
    wps = {mm: photon_mass_massive(mm) for mm in (1.0, 2.0, 5.0)}
    wp0 = 1.0 / np.sqrt(np.pi)
    out["ω_p(m=5) 与 e/√π 的偏差 < 5%"] = max(0.0, abs(wps[5.0] / wp0 - 1.0) - 0.05)
    for mm, v in wps.items():
        out[f"ω_p (m={mm})_magnitude"] = v
    out["e/√π_magnitude"] = wp0
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check_backreaction(), ensure_ascii=False, indent=2))
    print(json.dumps(check_massive(), ensure_ascii=False, indent=2))
