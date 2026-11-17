# -*- coding: utf-8 -*-
"""1+1D Kogut–Susskind 规范理论的精确对角化（含高斯定律投影与 E 谱截断）。

哈密顿量（连续时间、格点空间，a = 1）
------------------------------------
    H = (g²/2) Σ_x E_x²  +  Σ_x [ −t (χ†_x U_x χ_{x+1} + h.c.) ]  +  m Σ_x (−1)^x χ†_x χ_x

高斯定律（物理态条件）
----------------------
    E_x − E_{x−1} = ρ_x ,      ρ_x = χ†_xχ_x − ½(1 − (−1)^x)

在环上给定费米子占据后，电场由 `E_x = q + Σ_{y≤x} ρ_y` 唯一确定（q 是真空角标签）；
本模块取 q = 0 的一个扇区，并截断 `|E_x| ≤ Emax`。

做法
----
枚举满足 Σρ = 0 的费米子构型 → 算 E → 丢掉超截断的 → 在剩下的基上写 H（hooping 用
`χ†_x U_x χ_{x+1} + h.c.`，它同时把 E_x 抬/降 1，因而自动保持在物理子空间内）→ 稠密对角化。

N ≤ 10、Emax ≤ 3 时维数是几百，稠密足够。
"""
from __future__ import annotations

import numpy as np


def _rho(N: int, mask: int) -> np.ndarray:
    """ρ_x = n_x − ½(1 − (−1)^x)。"""
    r = np.empty(N)
    for x in range(N):
        n = (mask >> x) & 1
        r[x] = n - 0.5 * (1.0 - (1.0 if x % 2 == 0 else -1.0))
    return r


def _electric(N: int, mask: int, q: int = 0) -> np.ndarray:
    r = _rho(N, mask)
    E = np.empty(N)
    acc = 0.0
    for x in range(N):
        acc += r[x]
        E[x] = q + acc
    return E


def basis_and_hamiltonian(N: int, g: float, m: float, t: float = 1.0,
                          Emax: int = 2, q: int = 0, phi: float = 0.0):
    """构造物理子空间的基与 H（稠密厄米）。返回 (H, masks, E_stack)。

    `phi` = 穿过环的磁通（扭曲边界条件）：只有跨切口那一支 χ†_{N−1}χ_0 乘 e^{+iφ}、
    共轭支乘 e^{−iφ}，其余键不变。φ ≠ 0 时 H 是复厄米（dtype 跟着变），
    与 `ks_mps.mpo_ks(..., phi=φ)` 逐元素一致（P37 判据 ①）。
    """
    masks, Es = [], []
    for mask in range(1 << N):
        r = _rho(N, mask)
        if abs(np.sum(r)) > 1e-9:          # 环上总电荷必须为零
            continue
        E = _electric(N, mask, q)
        if np.max(np.abs(E)) > Emax + 1e-9:
            continue
        masks.append(mask)
        Es.append(E)
    index = {mk: i for i, mk in enumerate(masks)}
    dim = len(masks)
    H = np.zeros((dim, dim), dtype=complex if phi else float)
    zph = np.exp(1j * phi) if phi else 1.0
    zmh = np.exp(-1j * phi) if phi else 1.0

    for i, mask in enumerate(masks):
        E = Es[i]
        diag = 0.5 * g * g * float(np.sum(E * E))
        for x in range(N):
            diag += m * ((-1.0) ** x) * ((mask >> x) & 1)
        H[i, i] += diag

    # 跃迁：χ†_x U_x χ_{x+1} + h.c.（两种方向都写，保证厄米）
    for i, mask in enumerate(masks):
        for x in range(N):
            y = (x + 1) % N
            for (a, b) in ((x, y), (y, x)):
                if ((mask >> b) & 1) and not ((mask >> a) & 1):
                    new = mask ^ (1 << a) ^ (1 << b)
                    j = index.get(new)
                    if j is None:
                        continue
                    # 费米子符号：沿链的 Jordan–Wigner（与 E 无关）
                    lo, hi = (a, b) if a < b else (b, a)
                    between = bin(mask & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))).count("1")
                    sign = -1.0 if between % 2 else 1.0
                    if x == N - 1:                       # 跨切口键：带磁通相位
                        sign = sign * (zph if (a, b) == (N - 1, 0) else zmh)
                    H[i, j] += -t * sign
    return H, masks, np.array(Es)


def spectrum(N: int, g: float, m: float, t: float = 1.0, Emax: int = 2, q: int = 0,
             phi: float = 0.0):
    H, masks, Es = basis_and_hamiltonian(N, g, m, t, Emax, q, phi)
    w = np.linalg.eigvalsh(H)
    return w, masks, Es


def check_ks(N: int = 8, g: float = 1.0, m: float = 0.0) -> dict:
    """P34：Kogut–Susskind 精确对角化——高斯定律投影、厄米性、截断收敛、质量隙。"""
    out = {}
    H, masks, Es = basis_and_hamiltonian(N, g, m, Emax=2)
    out["维数 > 0"] = 0.0 if len(masks) > 0 else 1.0
    out["H 厄米（‖H−Hᵀ‖）"] = float(np.abs(H - H.T).max())

    # 高斯定律在**每个**基矢上成立：E_x − E_{x−1} = ρ_x
    gauss = 0.0
    for i, mask in enumerate(masks):
        E = Es[i]
        r = _rho(N, mask)
        gauss = max(gauss, float(np.abs((E - np.roll(E, 1)) - r).max()))
    out["高斯定律逐态成立"] = gauss

    # 截断收敛：Emax = 1, 2, 3 的最低两个本征值
    lows = []
    for em in (1, 2, 3):
        w, _, _ = spectrum(N, g, m, Emax=em)
        if len(w) >= 2:
            lows.append((em, float(w[0]), float(w[1]), len(w)))
    out["截断收敛 ΔE0(Emax1→2)_magnitude"] = abs(lows[1][1] - lows[0][1]) if len(lows) > 1 else 1.0
    out["截断收敛（ΔE0: Emax3−2）"] = abs(lows[2][1] - lows[1][1]) if len(lows) > 2 else 1.0
    out["截断饱和（Emax3 与 2 维数相同）"] = 0.0 if (len(lows) > 2 and lows[2][3] == lows[1][3]) else 1.0
    out["质量隙 E1−E0（m=%.2f）_magnitude" % m] = (lows[-1][2] - lows[-1][1]) if lows else float("nan")
    for em, e0, e1, d in lows:
        out[f"Emax={em} 维数_magnitude"] = float(d)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check_ks(), ensure_ascii=False, indent=2))
