# -*- coding: utf-8 -*-
"""磁通（扭曲边界）下的光子/弦态分辨，与电场关联长度的修正（P37）。

两个问题
--------
P36 留下的两件事：

  (a) `gap/e` 在 x ≥ 0.3 饱和（x = 0.4 时 N = 48/60 都是 0.657），但 x ≤ 0.2 不饱和
      （漂移 0.07 且不随 N 减小）。怀疑低激发是**弦态**（局域物体）而不是光子。
  (b) `⟨E_xE_0⟩_conn` 有长程底，ξ 拟合落到边界（给出 1/ξ ≈ 2.0 而 m_γ ≈ 0.06–0.26）。

做法 A：加磁通，看 gap(φ) 怎么动
--------------------------------
环上穿过磁通 φ 就是扭曲边界条件：跨切口那一支 χ†_{N−1}χ_0 得 e^{+iφ}、其共轭得
e^{−iφ}（`ks_mps.mpo_ks(..., phi=φ)`；φ ≠ 0 时 H 复厄米，DMRG 全部走复数分支）。判据：
  · 电荷 q 的**离域粒子**：`gap(φ)² = m² + q²(φ/N)²`，随 φ 二次上升，系数 q²（q = 1 时是 1）；
  · **局域弦态**（尺寸 ℓ ≪ N）：磁通只能靠绕环一周的虚过程进来 ⇒ 依赖 ~e^{−N/ℓ}，
    对 N = 24/32 是 ≤ 1e-3 的量级；实测若 ≥ 1e-2 就直接排除弦态。

单位约定（P36 已标定，这里再用自由极限验一次）：t = 0.5、a = 1、e² = 2g²t ⇒ e = g；
自由有质量极限下 N = 8/12 的稠密 ED 精确给出 `gap(φ) = 2√(m² + sin²(φ/N))`（1e-14），
即 **φ 对应的动量就是 φ/N、光速 = 1**，二次系数 1 就是 1²。这条解析式进了 P37 判据。

做法 B：逐点减 ⟨E_x⟩ 的连通关联
-------------------------------
① `⟨E_x⟩` 逐点算（E_x = Σ_{y≤x}ρ_y），不用 cell 平均；
② `C(r) = ⟨δE_x δE_{x+r}⟩`，x 取遍环平均，只取同奇偶的 r；
③ `C(r) = A(e^{−r/ξ} + e^{−(N−r)/ξ})` 取 ξ，`m_γ^corr = 1/ξ`。

关键实现细节（P36 的 ξ 之所以错，一半在这里）：环上的基态**不是**平移不变的
——交错费米子的 CDW 在切口处有缺陷（N=12、x=0.4 实测 ⟨ρ_x⟩ 从接缝的 ±0.36 涨到
中间的 ±0.44），所以 `⟨ρ_aρ_b⟩` 不能只按 (起点宇称, 距离) 推。本模块用
`ks_mps.corr_rho_from` 对**每个起点**各扫一遍，拿到**全部** N² 个两点函数，再累加成
`⟨δE_xδE_y⟩`。这样得到的 C(r) 关于 r ↔ N−r 对称、体相值与 N 无关（C(0) 在 N=24/32
上差 1.4%）；按"只依赖距离"硬推会得到一条假的线性斜坡（行和 Σ_b D(a,b) ≠ 0）。

实测结论（本机，χ = 96）
------------------------
**A（磁通）**：见 `FLUX_GAP_SCAN`（x = 0.4/0.2、N = 24/32、φ = 0/π2/π）。
  · **弦态被排除**：gap(π)/gap(0) − 1 = +32.4%/+8.7%（x=0.4，N=24/32）、
    +386.7%/+195.1%（x=0.2），比弦态能给的 ≲1e-3 大两三个量级；
  · **但也不是自由光子**：二次系数 b = Δgap²/(φ/N)² 实测 1.20–3.94（自由粒子 = 1）。
    x = 0.2 时 b = 3.942/3.941（N=24/32，**完全一致**，≈ 4 = 2²，像电荷 2 的模）；
    x = 0.4 时 b 随 N 漂（3.49 → 2.20），且 Δgap 更接近**线性**于 |φ|/N
    （斜率 0.470/0.563 与 0.209/0.224，同一 N 下两点差 19%/7%）。
  一句话：主导激发是**离域粒子型**（对 φ 敏感、弦态被排除），但既不是自由光子、
  也不是弦。

**B（关联）**：见 `CORR_SCAN`。C(r) 是干净的对称衰减曲线（体相 C(0)=0.1967/0.1939
（x=0.4，N=24/32）、0.2629/0.2649（x=0.2）），但它**不是单指数**：局部有效质量
m_eff(r) = ln[C(r)/C(r+2)]/2 从 0.63（r=1）单调降到 0.03（r=8），本底
C(N/2)/C(0) = 9.5%/6.7%（x=0.4）、23.5%/21.3%（x=0.2）。于是 1/ξ 有口径依赖：

    窗口 [2,6]   [2,8]   [4,10]       （同奇偶、不含 r=0 的 UV 掉头）
x=0.4,N=24  0.2431   0.2189  0.1562     gap = 0.2276  ⇒  +6.8% / −3.8% / −31.4%
x=0.4,N=32  0.2676   0.2366  0.1516     gap = 0.2527  ⇒  +5.9% / −6.4% / −40.0%
x=0.2,N=24  0.1509   0.1409  0.1099     gap = 0.0525  ⇒ +188%  / +169%  / +110%
x=0.2,N=32  0.1382   0.1248  0.0912     gap = 0.0658  ⇒ +110%  / +89%   / +39%

**任务 B 的硬指标（< 5%）只在窄口径下成立**：x=0.4 的近距窗口给 +6.8%/+5.9%，
[2,8] 给 −3.8%/−6.4%，窗口一拉到远距就掉到 −30%…−40%。所以诚实的说法是
"x=0.4 上两个观测量一致到 **4%–7%**（近距窗口，两个 N 一致），窗口系统差 ±20%；
x=0.2 上**不一致**（1.5–2.7 倍），说明那里的低激发不是电场算符能造出来的那个态"。
窗口一远 1/ξ 就变小，正是 C(r) 在 r ≳ 6 处被环上本底压平的后果（本底 ∝ e^{−N/2ξ}，
要它降到 ≲1e-3 得 N ≈ 60–80）。详见 OPERATOR_STATUS.md §6quindecies。
"""
from __future__ import annotations

import numpy as np

from . import ks_mps as _mps

__all__ = [
    "free_massive_gap",
    "gap_flux",
    "curvature_coeff",
    "string_signature",
    "corr_density_all",
    "corr_electric_connected",
    "xi_from_decay",
    "FLUX_GAP_SCAN",
    "FLUX_GAP_SMALL",
    "CORR_SCAN",
    "check_ks_photon",
]

# (x, N, φ/π, gap/e)：χ = 96、pen = 1、m = 0、seed = 5。复现见根目录 `ks_photon_scan.py`
# （N=24 单点 46–314 s、N=32 单点 104–469 s；φ ≠ 0 是复厄米 ⇒ 慢 2–3 倍）
FLUX_GAP_SCAN = (
    (0.40, 24, 0.0, 0.568882),
    (0.40, 24, 0.5, 0.645837),
    (0.40, 24, 1.0, 0.753042),
    (0.20, 24, 0.0, 0.262278),
    (0.20, 24, 0.5, 0.700700),
    (0.20, 24, 1.0, 1.276522),
    (0.40, 32, 0.0, 0.631774),
    (0.40, 32, 0.5, 0.657426),
    (0.40, 32, 1.0, 0.686818),
    (0.20, 32, 0.0, 0.329106),
    (0.20, 32, 0.5, 0.587980),
    (0.20, 32, 1.0, 0.971290),
)

# (x, N, φ/π, gap/e)：N = 12 的小表，专给自检现场复现（每点 1–4 s）
FLUX_GAP_SMALL = (
    (0.40, 12, 0.0, 0.371983),
    (0.40, 12, 0.5, 0.744464),
    (0.40, 12, 1.0, 1.262320),
    (0.20, 12, 0.0, 0.170126),
    (0.20, 12, 0.5, 1.313548),
    (0.20, 12, 1.0, 2.547564),
)

# (x, N, C(0), C(2), C(4), C(6), C(8), 1/ξ@[2,6], 1/ξ@[2,8], 1/ξ@[4,10], gap)：χ = 96
CORR_SCAN = (
    (0.40, 24, 0.196748, 0.056399, 0.032743, 0.024208, 0.020648,
     0.243100, 0.218900, 0.156200, 0.227553),
    (0.20, 24, 0.262910, 0.117444, 0.087317, 0.073568, 0.066412,
     0.150900, 0.140900, 0.109900, 0.052456),
    (0.40, 32, 0.193940, 0.053156, 0.028993, 0.020044, 0.016146,
     0.267600, 0.236600, 0.151600, 0.252710),
    (0.20, 32, 0.264893, 0.119144, 0.088452, 0.073958, 0.065860,
     0.124700, 0.124800, 0.091200, 0.065821),
)


# ==============================================================================
# 1. 磁通：解析基准、实测、两个判据量
# ==============================================================================
def free_massive_gap(N: int, m: float, phi: float = 0.0, t: float = 0.5) -> float:
    """自由有质量极限（g = 0）的解析 gap：`2√(m² + sin²(φ/N))`（2t = 1 的单位）。

    半满自由链的单粒子色散 ε(k) = √(m² + (2t cos k)²)（t = 0.5 ⇒ 2t = 1），粒子–空穴
    对在同一 k 上激发 ⇒ gap = 2ε；磁通把 k = π/2 移到 π/2 + φ/N。N = 8 与 12 的稠密
    ED 与这条式子吻合到 1e-14（P37 判据）。

    注意 N ≡ 2 (mod 4) 时动量格点整体平移 π：实测 N = 10 的 gap(0) 等于本式在
    φ = π 处的值、gap(π) 等于 φ = 0 处的值。所以判据只用 N ≡ 0 (mod 4)。
    """
    return float(2.0 * np.sqrt(m * m + np.sin(phi / N) ** 2))


def gap_flux(N: int, x: float, phi: float = 0.0, chi: int = 96, sweeps: int = 12,
             seed: int = 5, m: float = 0.0) -> dict:
    """在磁通 φ 下算一次 gap：返回 {N, x, phi, E0, E1, gap, gap_over_e, e, bond}。

    `x` 就是 g（本包单位下 x = ea = g，t = 0.5）。φ ≠ 0 时 H 复厄米，DMRG 走复数分支。
    """
    r = _mps.dmrg_gap(N, x, m, chi=chi, sweeps=sweeps, seed=seed, phi=phi)
    return {"N": int(N), "x": float(x), "phi": float(phi), "E0": r["E0"], "E1": r["E1"],
            "gap": r["gap"], "gap_over_e": r["gap_over_e"], "e": r["e"],
            "bond": r["bond"]}


def curvature_coeff(g0: float, g1: float, phi: float, N: int) -> float:
    """二次系数 b = (gap(φ)² − gap(0)²)/(φ/N)²；自由粒子（光速 1、电荷 1）b = 1。"""
    u = (phi / N) ** 2
    return float((g1 * g1 - g0 * g0) / u) if u > 0 else float("nan")


def string_signature(g0: float, g1: float) -> float:
    """弦态判据用的相对抬升：gap(π)/gap(0) − 1（局域物体应 ~e^{−N/ℓ} ≲ 1e-3）。"""
    return float(g1 / g0 - 1.0) if g0 > 0 else float("nan")


# ==============================================================================
# 2. 逐点减除的电场连通关联
# ==============================================================================
def corr_density_all(A, ops=None):
    """⟨ρ_aρ_b⟩ 的全部 N² 个值（每个起点各扫一次，O(N²χ³)）。

    环上基态只有"平移 2 格"的对称性，接缝处还有缺陷，所以**不能**假设
    ⟨ρ_aρ_b⟩ 只依赖 (a 的宇称, b−a)；这里逐个起点取精确值。
    """
    N = len(A)
    if ops is None:
        ops = [_mps._rho_op(x) for x in range(N)]
    R = np.zeros((N, N))
    for a in range(N):
        c = _mps.corr_rho_from(A, ops, a)
        R[a, a:a + len(c)] = c
        R[a:a + len(c), a] = c
    return R


def corr_electric_connected(rho, R, N):
    """C(r) = ⟨δE_x δE_{x+r}⟩，逐点减 ⟨E_x⟩、对 x 平均（不做 cell 平均）。

    E_x = Σ_{y≤x}ρ_y ⇒ ⟨δE_xδE_y⟩ = Σ_{a≤x}Σ_{b≤y}D(a,b)，D(a,b) = ⟨ρ_aρ_b⟩−⟨ρ_a⟩⟨ρ_b⟩；
    二维前缀和一次算完。行和 Σ_b D(a,b) = ⟨ρ_aΣρ⟩ − ⟨ρ_a⟩⟨Σρ⟩ = 0（Q = 0 扇区）
    ⇒ 不会出现线性斜坡，C(r) 关于 r ↔ N−r 对称。
    """
    rho = np.asarray(rho, dtype=float)
    D = np.asarray(R, dtype=float) - np.outer(rho, rho)
    P = np.zeros((N + 1, N + 1))
    P[1:, 1:] = np.cumsum(np.cumsum(D, axis=0), axis=1)
    E2 = P[1:, 1:]                          # E2[x,y] = ⟨δE_xδE_y⟩
    return np.array([float(np.mean([E2[x, (x + r) % N] for x in range(N)]))
                     for r in range(N)])


def xi_from_decay(C, N: int, parity: int = 0, window=(2, 6),
                  with_c0: bool = False, npts: int = 8000) -> dict:
    """从 C(r) 取衰减长度：C(r) = [c₀ +] A(e^{−r/ξ} + e^{−(N−r)/ξ})。

    `window` 是**绝对距离**（格点单位，如 (2, 6)），只取 r ≡ parity (mod 2) 的点，
    这样不同 N 用的是同一段物理距离（用 N 的分数当窗口会让 N=24 与 32 分别落在
    本底与体相上，ξ 差一倍）。扫 ξ、线性解 A（和 c₀），取相对偏差最小、A > 0 的解。
    """
    C = np.asarray(C, dtype=float)
    rs = np.arange(parity, N, 2)
    use = rs[(rs >= window[0]) & (rs <= min(window[1], N // 2))]
    if len(use) < 3:
        use = rs[1:min(len(rs), 6)]
    ys = C[use]
    best = None
    for xi in np.exp(np.linspace(np.log(0.5), np.log(2.0 * N), npts)):
        f = np.exp(-use / xi) + np.exp(-(N - use) / xi)
        if with_c0:
            X = np.vstack([np.ones_like(f), f]).T
            c, *_ = np.linalg.lstsq(X, ys, rcond=None)
            c0, A = float(c[0]), float(c[1])
        else:
            c0, A = 0.0, float(np.dot(f, ys) / np.dot(f, f))
        if A <= 0.0:
            continue
        model = c0 + A * f
        dev = float(np.max(np.abs(model - ys)) / max(np.max(np.abs(model)), 1e-300))
        if best is None or dev < best[3]:
            best = (A, float(xi), c0, dev)
    if best is None:
        return {"A": float("nan"), "xi": float("nan"), "m": float("nan"),
                "c0": float("nan"), "dev": float("nan"), "window": (0, 0), "n": 0}
    return {"A": best[0], "xi": best[1], "m": 1.0 / best[1], "c0": best[2],
            "dev": best[3], "window": (int(use[0]), int(use[-1])), "n": int(len(use))}


# ==============================================================================
# 3. 自检（P37）
# ==============================================================================
def _ed_module():
    from . import ks_ed
    return ks_ed


def _diag(x, N, phi):
    """从 FLUX_GAP_SCAN 取 (gap, gap/e)。"""
    for (xx, nn, pp, v) in FLUX_GAP_SCAN:
        if abs(xx - x) < 1e-9 and nn == N and abs(pp - phi) < 1e-6:
            return v * x, v
    raise KeyError((x, N, phi))


def _check_corr_rho_from(N=10, g=0.6):
    """`corr_rho_from` 的全部 (a,b) 与稠密 ED 的 ⟨ρ_aρ_b⟩ 对拍。"""
    ked = _ed_module()
    H, masks, _ = ked.basis_and_hamiltonian(N, g, 0.0, t=0.5, Emax=N + 2)
    _w, v = np.linalg.eigh(H)
    p = v[:, 0] ** 2
    rmat = np.array([[(int((mk >> x) & 1)) - 0.5 * (1.0 - ((-1.0) ** x)) for mk in masks]
                     for x in range(N)])
    R_ed = (rmat * p) @ rmat.T
    r = _mps.dmrg_ks(N, g, chi=96, sweeps=8, seed=5)
    return float(np.abs(corr_density_all(r["A"]) - R_ed).max())


def check_ks_photon() -> dict:
    """P37：磁通分辨光子/弦态 + 逐点减除的电场关联（实测值见模块文档）。

    **结构判据（进残差，全部机器精度）**
      ① 扭曲 MPO vs 扭曲 `ks_ed` 的稠密 H（N=6/8、φ=0/π2/π、m=0/0.4）+ 厄米性
      ② φ = 2π 与 φ = 0 的 H 逐元素相同（周期 2π）
      ③ 自由极限解析色散 `2√(m²+sin²(φ/N))`（g=0、m=0.3、N=8/12、φ=0/π2/π）
         —— 把"φ ↔ 动量 φ/N、光速 1"钉死，也是二次系数判据的锚
      ④ 扭曲 DMRG vs 扭曲 ED：E0 与 gap（N=8 g=1.0、N=10 g=0.5，φ=π/2、π）
      ⑤ 电荷扇区 ⟨Q²⟩（N=8、x=1.0、φ=π/2）
      ⑥ 小表现场复现（N=12、x=0.4/0.2、φ=π/2，判据 ≤1e-6；表存 6 位小数）
      ⑦ `corr_rho_from` 对**全部** (a,b) 与稠密 ED 一致（新公开 API 真的被调用）
      ⑧ 点向减除的行和 Σ_b D(a,b)（到机器精度，否则 ξ 一定被斜坡污染）
    **物理判据（阈值写死在实测表上，可稳定复现）**
      ⑨ 弦态排除：min over 4 点 [gap(π)/gap(0) − 1] ≥ 0.02（实测 0.041）
      ⑩ 自由光子律不成立：min b(φ=π/2) ≥ 1.3（实测 2.20；自由粒子恰为 1）
      ⑪ 关联函数体相值与 N 无关：max|C(0;24)−C(0;32)|/C(0) ≤ 0.05（实测 0.014/0.008）
      ⑫ 两个观测量在饱和点一致：x=0.4、窗口 [2,6] 上 |m_corr/m_gap−1| ≤ 0.15
         （实测 +6.8%（N=24）/ +5.9%（N=32）；换成 [2,8] 是 −3.8%/−6.4%，
          拉到 [4,10] 掉到 −31%/−40%——窗口系统差写在 `_magnitude` 里）
    **物理结论（_magnitude，只报数不判定）**
      · gap(φ) 表、二次系数 b、弦态判据值、三种窗口下的 1/ξ 与相对差
    """
    out = {}
    ked = _ed_module()

    # ① 扭曲 MPO vs 扭曲 ED
    worst = 0.0
    for N in (6, 8):
        for (g, m) in ((1.0, 0.0), (0.7, 0.4)):
            for phi in (0.0, np.pi / 2, np.pi):
                H_ed, masks, _ = ked.basis_and_hamiltonian(N, g, m, t=0.5, Emax=N + 2,
                                                           phi=phi)
                mpo, D = _mps.mpo_ks(N, g, m, t=0.5, phi=phi)
                H_mpo = _mps.mpo_dense(mpo, D)
                worst = max(worst,
                            float(np.abs(H_mpo[np.ix_(masks, masks)] - H_ed).max()),
                            float(np.abs(H_mpo - H_mpo.conj().T).max()))
    out["扭曲 MPO vs 稠密 ED 的 H（N=6/8，φ=0/π2/π，m=0/0.4）"] = worst

    # ② 周期 2π
    H0 = _mps.mpo_dense(_mps.mpo_ks(6, 1.0, 0.0, t=0.5, phi=0.0)[0], 8)
    H2 = _mps.mpo_dense(_mps.mpo_ks(6, 1.0, 0.0, t=0.5, phi=2 * np.pi)[0], 8)
    out["φ=2π 与 φ=0 的 H 相同（N=6）"] = float(np.abs(H0 - H2).max())

    # ③ 自由极限解析色散
    worst = 0.0
    for N in (8, 12):
        for phi in (0.0, np.pi / 2, np.pi):
            H, _masks, _ = ked.basis_and_hamiltonian(N, 0.0, 0.3, t=0.5, Emax=N + 2,
                                                      phi=phi)
            w = np.linalg.eigvalsh(H)
            worst = max(worst, abs(float(w[1] - w[0]) - free_massive_gap(N, 0.3, phi)))
    out["自由极限 2√(m²+sin²(φ/N))（g=0,m=0.3,N=8/12）"] = worst

    # ④ 扭曲 DMRG vs 扭曲 ED
    e0w = gapw = 0.0
    for (N, g, chi) in ((8, 1.0, 64), (10, 0.5, 96)):
        for phi in (np.pi / 2, np.pi):
            H, _masks, _ = ked.basis_and_hamiltonian(N, g, 0.0, t=0.5, Emax=N + 2, phi=phi)
            w = np.linalg.eigvalsh(H)
            r = _mps.dmrg_gap(N, g, chi=chi, sweeps=10, seed=5, phi=phi)
            e0w = max(e0w, abs(r["E0"] - float(w[0])))
            gapw = max(gapw, abs(r["gap"] - float(w[1] - w[0])))
    out["扭曲 DMRG vs 扭曲 ED：E0（N=8/10，φ=π2/π）"] = e0w
    out["扭曲 DMRG vs 扭曲 ED：gap（同上）"] = gapw

    # ⑤ 电荷扇区
    r8 = _mps.dmrg_ks(8, 1.0, chi=64, sweeps=10, seed=5, phi=np.pi / 2)
    out["电荷扇区 ⟨Q²⟩（N=8,x=1.0,φ=π2）"] = abs(_mps.energy(_mps._q2_mpo(8), r8["A"]))

    # ⑥ 小表现场复现
    worst = 0.0
    for (x, N, ph, val) in FLUX_GAP_SMALL:
        if N == 12 and abs(ph - 0.5) < 1e-9:
            g = gap_flux(N, x, ph * np.pi, chi=96, sweeps=12)
            worst = max(worst, abs(g["gap_over_e"] - val))
    out["表内现场复现（N=12，x=0.4/0.2，φ=π2）"] = worst if worst else 1.0

    # ⑦ corr_rho_from vs ED
    out["corr_rho_from vs 稠密 ED 全对 ⟨ρ_aρ_b⟩（N=10,x=0.6）"] = _check_corr_rho_from()

    # ⑧ 行和
    r12 = _mps.dmrg_ks(12, 0.4, chi=96, sweeps=10, seed=5)
    ops = [_mps._rho_op(k) for k in range(12)]
    rho12 = _mps.one_site_all(r12["A"], ops)
    R12 = corr_density_all(r12["A"], ops)
    D12 = R12 - np.outer(rho12, rho12)
    out["点向减除后的行和 max_a|Σ_b D(a,b)|（N=12,x=0.4）"] = float(np.abs(D12.sum(axis=1)).max())
    C12 = corr_electric_connected(rho12, R12, 12)

    pts = ((0.40, 24), (0.40, 32), (0.20, 24), (0.20, 32))

    # ⑨ 弦态排除
    rises = [string_signature(_diag(x, N, 0.0)[0], _diag(x, N, 1.0)[0]) for (x, N) in pts]
    out["弦态判据 min[gap(π)/gap(0)−1] ≥ 0.02（实测 0.041）"] = max(0.0, 0.02 - min(rises))

    # ⑩ 二次系数
    bs = [curvature_coeff(_diag(x, N, 0.0)[0], _diag(x, N, 0.5)[0], np.pi / 2, N)
          for (x, N) in pts]
    out["自由光子律不成立 min b(φ=π2) ≥ 1.3（实测 2.20；自由粒子 = 1）"] = max(
        0.0, 1.3 - min(bs))

    # ⑪ 体相 C(0) 的 N 无关性
    c0v = {(float(r[0]), int(r[1])): float(r[2]) for r in CORR_SCAN}
    dev_bulk = max(abs(c0v[(x, 24)] - c0v[(x, 32)]) / c0v[(x, 24)] for x in (0.40, 0.20))

    # ⑫ 两个观测量在饱和点的一致性（[2,6] 窗口）
    rel_near = [abs(r[7] / r[10] - 1.0) for r in CORR_SCAN if abs(r[0] - 0.40) < 1e-9]
    out["m_corr 与 m_gap 在 x=0.4 一致（[2,6] 窗口，|rel| ≤ 0.15，实测 0.068/0.059）"] = max(
        0.0, max(rel_near) - 0.15)

    # 物理结论
    for ((x, N), rr, bb) in zip(pts, rises, bs):
        out["gap(π)/gap(0)−1（x=%.2f,N=%d）_magnitude" % (x, N)] = rr
        out["二次系数 b(φ=π2)（x=%.2f,N=%d）_magnitude" % (x, N)] = bb
        out["gap/e（x=%.2f,N=%d,φ=0）_magnitude" % (x, N)] = _diag(x, N, 0.0)[1]
        out["gap（x=%.2f,N=%d,φ=π）_magnitude" % (x, N)] = _diag(x, N, 1.0)[0]
    out["体相 C(0) 的 N 相对差_magnitude"] = dev_bulk
    for (x, N, C0v, _c2, _c4, _c6, _c8, m26, m28, m410, gap) in CORR_SCAN:
        e = float(x)
        tag = "x=%.2f,N=%d" % (x, N)
        out["C(0)（%s）_magnitude" % tag] = C0v
        out["1/(ξe)@[2,6]（%s）_magnitude" % tag] = m26 / e
        out["m_corr/m_gap−1@[2,6]（%s）_magnitude" % tag] = m26 / gap - 1.0
        out["m_corr/m_gap−1@[2,8]（%s）_magnitude" % tag] = m28 / gap - 1.0
        out["m_corr/m_gap−1@[4,10]（%s）_magnitude" % tag] = m410 / gap - 1.0
        out["gap/e（%s）_magnitude" % tag] = gap / e
    out["C(0)（N=12,x=0.4）_magnitude"] = float(C12[0])
    out["FLUX_GAP_SCAN 行数_magnitude"] = float(len(FLUX_GAP_SCAN))
    out["FLUX_GAP_SMALL 行数_magnitude"] = float(len(FLUX_GAP_SMALL))
    out["CORR_SCAN 行数_magnitude"] = float(len(CORR_SCAN))
    return out
