# -*- coding: utf-8 -*-
"""矩阵自由的 Lanczos：1+1D Kogut–Susskind 的质量隙与连续极限。

为什么需要它
------------
`ks_ed.py` 走稠密对角化，N ≤ 14；质量隙在 N = 6/8/10/12/14 上给出
1.96/0.75/1.32/0.91/1.13，来回振荡——因为 N、E 谱截断、物理单位三项外推一项都没做。
本模块不物化任何矩阵：基矢是枚举出的费米子构型（mask），H 的 matvec 由对角项 +
每条键的跃迁表直接算，本征值用带全重正交的 Lanczos（纯 numpy），于是 N 推到 24。

哈密顿量与约定（与 `ks_ed.py` 逐项一致）
----------------------------------------
    H = (g²/2) Σ_x E_x² + t Σ_x (χ†_x U_x χ_{x+1} + h.c.) + m Σ_x (−1)^x n_x
    E_x − E_{x−1} = ρ_x ,   ρ_x = n_x − ½(1 − (−1)^x)
基矢 = 满足 Σρ = 0 且 |E_x| ≤ Emax 的所有构型（环上 q = 0 扇区）。

物理单位（连续极限的唯一无量纲参数）
------------------------------------
交错费米子的动能归一化给 t = 1/(2a)；电场项 (g²/2)ΣE² 对上 ∫dx ½E² 给 g² = e²a，
于是 e² = 2g²t，而「以 e 为单位」的隙只依赖无量纲的

    x ≡ e a = g / √(2t)                （t = 0.5 时 x = g）

    gap(N, x) / e  ──(N→∞, x→0)──▶  1/√π = 0.5641895

两者都是本模块的**独立标定**，不是假设：强耦合极限 gap → g²/2（P35 判据 ③）钉住
H 的归一化，自由极限 E0 = Σ 最低 N/2 个 ±2t|cos(k/2)|（判据 ②）钉住环上的 JW 符号。
连续极限本身见 `fit_continuum_gap`（P35 的 _magnitude 输出）。

尺寸与代价（本机实测，`table_limit` 内用 mask→下标直查表，否则 searchsorted）
--------------------------------------------------------------------------
    N    :  18      20       22       24       26
    dim  :  4.5e4   1.6e5    6.0e5    2.2e6    8.2e6
    1 次 matvec : 12 ms  60 ms  0.4 s  1.6 s   ~6 s
    Lanczos 200 步 : 2.5 s  12 s    80 s    313 s    —
N ≥ 28 时 dim ≈ 3e7（≈ 4^N/√(πN/2)），单个矢量 240 MB，200 步全重正交要约 48 GB，
不再可行——这就是本模块的边界。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "basis_1d",
    "KogutSusskind1D",
    "lanczos_lowest",
    "gap_lanczos",
    "GAP_SCAN",
    "fit_continuum_gap",
    "check_ks_continuum",
]

_U64 = np.uint64
SQRT_PI_INV = float(1.0 / np.sqrt(np.pi))          # 0.5641895…，Schwinger 光子质量 e/√π


def _popcount64(x) -> np.ndarray:
    """uint64 数组的按位 1 计数（SWAR，不依赖 numpy 版本的 bit_count）。"""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    x = x - ((x >> _U64(1)) & _U64(0x5555555555555555))
    x = (x & _U64(0x3333333333333333)) + ((x >> _U64(2)) & _U64(0x3333333333333333))
    x = (x + (x >> _U64(4))) & _U64(0x0F0F0F0F0F0F0F0F)
    return (x * _U64(0x0101010101010101)) >> _U64(56)


def basis_1d(N: int, Emax: int = 2, q: int = 0):
    """物理子空间的基矢：返回 (masks, sumE2)。

    masks 升序（int64，bit x = 格点 x 占据），sumE2 = Σ_x E_x² 逐态。
    逐格点生长：x 偶时 ρ ∈ {0,1}，x 奇时 ρ ∈ {−1,0}；每步要求 |E_x| ≤ Emax，
    收尾要求 Σρ = 0（等价于 E_{N−1} = q，环上总电荷为零）。
    维数增长律实测 ≈ C(N, N/2) ≈ 4^N/√(πN/2)：Emax ≥ 2 后截断只削掉几个百分点
    （走子只在偶格上升、奇格下降，|E| ≤ 2 天然贴合半填充）。
    """
    masks = np.zeros(1, dtype=np.int64)
    ecur = np.full(1, q, dtype=np.int64)
    se2 = np.zeros(1, dtype=np.int64)
    for x in range(N):
        r0, r1 = (0, 1) if x % 2 == 0 else (-1, 0)
        e0 = ecur + r0
        e1 = ecur + r1
        k0 = np.abs(e0) <= Emax
        k1 = np.abs(e1) <= Emax
        masks = np.concatenate((masks[k0], (masks | np.int64(1 << x))[k1]))
        ecur = np.concatenate((e0[k0], e1[k1]))
        se2 = np.concatenate((se2[k0] + e0[k0] ** 2, se2[k1] + e1[k1] ** 2))
    keep = ecur == q                      # Σρ = 0
    masks, se2 = masks[keep], se2[keep]
    order = np.argsort(masks, kind="stable")
    return masks[order], se2[order]


class KogutSusskind1D:
    """1+1D Kogut–Susskind 哈密顿量，矩阵自由表示（对角项 + 跃迁表）。

    只存 O(dim) 的对角项与 O(dim·N) 的整数下标表，不存矩阵元；`matvec` 是
    N 条键上的 gather + scatter。厄米性由「每条键两端同时置位」自动成立：
    对一条键，源态与目标态都在 `src` 里，`y[s] += c·v[j]` 与 `y[j] += c·v[s]`
    是同一次循环里做的（c 对称）。
    """

    def __init__(self, N: int, g: float, m: float = 0.0, t: float = 0.5,
                 Emax: int = 2, q: int = 0, table_limit: int = 1 << 26):
        if N < 4 or N % 2:
            raise ValueError("N 必须是 ≥ 4 的偶数（半填充要求）")
        self.N, self.g, self.m, self.t, self.Emax, self.q = N, g, m, t, Emax, q
        self.masks, self.sumE2 = basis_1d(N, Emax, q)
        self.dim = int(self.masks.size)
        self.diag = 0.5 * g * g * self.sumE2.astype(float)
        if m != 0.0:
            occ = np.zeros(self.dim)
            for x in range(N):
                occ += ((-1.0) ** x) * ((self.masks >> x) & 1)
            self.diag = self.diag + m * occ
        # 跃迁表：每条键 (a, b) 的源下标、目标 mask、中间位掩码（JW 符号用）
        self.bonds = []
        for x in range(N):
            a, b = x, (x + 1) % N
            lo, hi = (a, b) if a < b else (b, a)
            mid = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
            sel = ((self.masks >> a) & 1) != ((self.masks >> b) & 1)
            src = np.nonzero(sel)[0].astype(np.int64)
            tgt = self.masks[src] ^ np.int64((1 << a) | (1 << b))
            self.bonds.append((src, tgt, mid))
        self.pos = None
        if (1 << N) <= table_limit:
            pos = np.full(1 << N, -1, dtype=np.int32)
            pos[self.masks] = np.arange(self.dim, dtype=np.int32)
            self.pos = pos

    def _lookup(self, tgt) -> np.ndarray:
        """目标 mask → 下标；不在截断基矢里给 −1（那一条跃迁被丢掉）。"""
        if self.pos is not None:
            return self.pos[tgt]
        idx = np.clip(np.searchsorted(self.masks, tgt), 0, self.dim - 1)
        return np.where(self.masks[idx] == tgt, idx, -1)

    def matvec(self, v):
        """y = H v（不物化矩阵；每条键一次 gather + 一次 scatter）。"""
        y = self.diag * v
        for src, tgt, mid in self.bonds:
            j = self._lookup(tgt)
            ok = j >= 0
            s, jj = (src[ok], j[ok]) if not ok.all() else (src, j)
            par = _popcount64((self.masks[s] & np.int64(mid)).astype(np.uint64))
            sign = 1.0 - 2.0 * (par & _U64(1)).astype(float)
            y[s] += (-self.t * sign) * v[jj]
        return y

    def rayleigh(self, v):
        """瑞利商与残差 ‖Hv − λv‖。"""
        hv = self.matvec(v)
        lam = float(np.dot(v, hv))
        return lam, float(np.linalg.norm(hv - lam * v))


def lanczos_lowest(op: KogutSusskind1D, k: int = 2, iters: int = 200,
                   seed: int = 7, max_vectors: int | None = None) -> dict:
    """带两遍全重正交的 Lanczos：返回最低 k 个 Ritz 值、矢量与残差。

    三对角块整体对角化后取最低 k 个；`resid` 是 ‖Hv − λv‖ 的实测值（保证谱值可信），
    `orth` 是 Ritz 矢的相互正交性误差。`max_vectors` 用来在 dim 大时限制内存
    （每个矢量 8·dim 字节），超出就提前停。
    """
    n = min(iters, op.dim)
    if max_vectors is not None:
        n = min(n, int(max_vectors))
    V = np.zeros((n, op.dim))
    alpha = np.zeros(n)
    beta = np.zeros(n)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=op.dim)
    v /= np.linalg.norm(v)
    V[0] = v
    used = 0
    for j in range(n):
        w = op.matvec(V[j])
        alpha[j] = float(np.dot(V[j], w))
        if j > 0:
            w = w - beta[j - 1] * V[j - 1]
        w = w - alpha[j] * V[j]
        for _ in range(2):                      # 两遍全重正交
            w -= V[: j + 1].T @ (V[: j + 1] @ w)
        b = float(np.linalg.norm(w))
        used = j + 1
        if b < 1e-12 or j == n - 1:
            break
        beta[j] = b
        V[j + 1] = w / b
    T = np.diag(alpha[:used])
    if used > 1:
        off = beta[: used - 1]
        T = T + np.diag(off, 1) + np.diag(off, -1)
    w, U = np.linalg.eigh(T)
    kk = min(k, used)
    vecs = V[:used].T @ U[:, :kk]
    resid = []
    for i in range(kk):
        hv = op.matvec(vecs[:, i])
        resid.append(float(np.linalg.norm(hv - w[i] * vecs[:, i])))
    return {"vals": w[:kk], "vecs": vecs, "resid": np.array(resid),
            "iters": used, "orth": float(np.abs(vecs.T @ vecs - np.eye(kk)).max())}


def gap_lanczos(N: int, g: float, m: float = 0.0, t: float = 0.5, Emax: int = 2,
                iters: int = 200, seed: int = 7, k: int = 2,
                max_vectors: int | None = None) -> dict:
    """最低两能级与质量隙（以 e = g√(2t) 为单位；x = ea = g/√(2t)）。"""
    op = KogutSusskind1D(N, g, m=m, t=t, Emax=Emax)
    out = lanczos_lowest(op, k=max(k, 2), iters=iters, seed=seed,
                         max_vectors=max_vectors)
    vals = out["vals"]
    e = g * np.sqrt(2.0 * t)
    gap = float(vals[1] - vals[0])
    return {"N": N, "g": g, "t": t, "Emax": Emax, "m": m, "dim": op.dim,
            "e": float(e), "x": float(e / (2.0 * t)) if e > 0.0 else 0.0,
            "E0": float(vals[0]),
            "E1": float(vals[1]), "gap": gap,
            "gap_over_e": (gap / e) if e > 0.0 else float("nan"),
            "resid": float(out["resid"].max()), "iters": out["iters"],
            "orth": out["orth"]}


# ==============================================================================
# 连续极限：实测网格与外推
# ==============================================================================
# (x, N, gap/e)：Emax = 2、m = 0、t = 0.5、200 步 Lanczos 的实测值。
# 复现：仓库根 `python ks_continuum_scan.py`（N = 24 的单点约 5 分钟）。
GAP_SCAN = (
    (0.30, 20, 0.378733880687),
    (0.30, 22, 0.950287780893),
    (0.30, 24, 0.431627062687),
    (0.40, 18, 0.894494206715),
    (0.40, 20, 0.514848210293),
    (0.40, 22, 0.786765451885),
    (0.40, 24, 0.569086775125),
    (0.50, 16, 0.556003006984),
    (0.50, 18, 0.793147712684),
    (0.50, 20, 0.617319647590),
    (0.50, 22, 0.734252319162),
    (0.50, 24, 0.653556116422),
    (0.70, 16, 0.698862541717),
    (0.70, 18, 0.751337406310),
    (0.70, 20, 0.720654798556),
    (0.70, 22, 0.735029558326),
    (0.70, 24, 0.725472485636),
    (1.00, 16, 0.793472163524),
    (1.00, 18, 0.795370363399),
    (1.00, 20, 0.792633283115),
    (1.00, 22, 0.791956819331),
    (1.00, 24, 0.790837303367),
)


def _limit_n_inf(ns):
    """固定 x 的 N→∞ 极限：a + b·(−1)^{N/2}·e^{−κN}，κ 一维扫描 + 线性最小二乘。

    半填充环上有零模时（N ≡ 0 mod 4）低谱结构与 N ≡ 2 mod 4 不同，实测到的主导
    有限尺寸效应就是这个「奇偶交替 + 指数衰减」；κ 不预设，由数据定。
    """
    n = np.array([p[0] for p in ns], float)
    v = np.array([p[1] for p in ns], float)
    osc = (-1.0) ** (n / 2.0)
    best = None
    for kap in np.linspace(0.0, 1.2, 241):
        A = np.vstack([np.ones_like(n), osc * np.exp(-kap * n)]).T
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        dev = float(np.max(np.abs(A @ c - v)))
        if best is None or dev < best[2]:
            best = (float(c[0]), float(kap), dev)
    return best


def fit_continuum_gap(points=None, n_min: int = 16, x_min: float = 0.5) -> dict:
    """把 gap/e 外推到 (N→∞, x→0)，返回连续极限与外推不确定度。

    两步，参数都从数据里定，不预设答案：

    ① 固定 x：`gap/e = a + b·(−1)^{N/2} e^{−κN}`（κ 扫描）。零模结构给奇偶交替，
       指数衰减率随物理体积 eL = x·N 走，所以实测 κ×x ≈ 0.2–0.4。
    ② 全局：`G(x) = G0 + A·x^p`，p 由数据网格搜索定。p 不是假设——它是「格点伪影
       随 x 掉多快」的实测结果（本机 4–5 个 x 给 p ≈ 1）。
    不确定度 = 以下几种 G0 的极差（模型依赖）：p 自由、p = 1、p = 1.5、p = 2、
    以及 `G0 + c1 x² + c2 x⁴` 多项式外推。

    `x_min` 默认 0.5：x ≤ 0.4 的点在本机 N ≤ 24 下**不可控**——零模奇偶振荡已经
    涨到 ±35%（x=0.4, N=18/20 实测 0.894/0.515），N→∞ 极限只是三四个振荡点的
    外插。那些点仍然记在 GAP_SCAN 里（是实测值），但默认不进入外推。
    """
    pts = GAP_SCAN if points is None else points
    pts = [(float(a), int(b), float(c)) for a, b, c in pts
           if int(b) >= n_min and float(a) >= x_min]
    if len(pts) < 3:
        return {"G0": float("nan"), "target": SQRT_PI_INV, "dev_rel": float("nan"),
                "spread": float("nan"), "p": float("nan"), "max_dev": float("nan"),
                "rows": [], "n_points": len(pts), "forms": []}
    groups = {}
    for x, n, v in pts:
        groups.setdefault(round(x, 9), []).append((n, v))
    rows = []
    for x in sorted(groups):
        a, kap, dev = _limit_n_inf(groups[x])
        rows.append({"x": x, "G": a, "kappa": kap, "dev": dev,
                     "N": [min(p[0] for p in groups[x]), max(p[0] for p in groups[x])]})
    xv = np.array([r["x"] for r in rows])
    gv = np.array([r["G"] for r in rows])

    def power_fit(p):
        A = np.vstack([np.ones_like(xv), xv ** p]).T
        c, *_ = np.linalg.lstsq(A, gv, rcond=None)
        return float(c[0]), float(np.max(np.abs(A @ c - gv)))

    best = None
    for p in np.linspace(0.4, 3.0, 261):
        g0, dev = power_fit(float(p))
        if best is None or dev < best[2]:
            best = (g0, float(p), dev)
    g0, p_hat, dev0 = best
    forms = [("p=%.2f（自由）" % p_hat, g0)]
    for pf in (1.0, 1.5, 2.0):
        forms.append(("p=%.1f" % pf, power_fit(pf)[0]))
    if len(xv) >= 4:
        A = np.vstack([np.ones_like(xv), xv ** 2, xv ** 4]).T
        c, *_ = np.linalg.lstsq(A, gv, rcond=None)
        forms.append(("c1x²+c2x⁴", float(c[0])))
    g0s = [g for _, g in forms]
    dev_all = max([dev0] + [r["dev"] for r in rows])
    return {"G0": g0, "p": p_hat, "target": SQRT_PI_INV,
            "dev_rel": abs(g0 - SQRT_PI_INV) / SQRT_PI_INV,
            "spread": max(g0s) - min(g0s), "max_dev": dev_all,
            "rows": rows, "forms": forms, "n_points": len(pts),
            "x_range": [float(xv.min()), float(xv.max())],
            "N_range": [int(min(p[1] for p in pts)), int(max(p[1] for p in pts))]}


# ==============================================================================
# P35：判据集合
# ==============================================================================
def _dense_module():
    """取 ks_ed（包内形态或同目录单文件形态）。"""
    try:
        from . import ks_ed
        return ks_ed
    except ImportError:                                  # pragma: no cover
        import ks_ed
        return ks_ed


def free_limit_energy(N: int, t: float = 0.5) -> float:
    """g = 0 时半填充环的基态能量 = 最低 N/2 个单粒子能级之和。

    单粒子谱 ε(k) = ±2t|cos(k/2)|，k = 4πj/N（**周期**边界；交错费米子的一格平移
    在环上给 PBC 而非 APBC，这是本模块与 `ks_ed.py` 共用的 JW 符号约定的直接后果）。
    """
    eps = np.array([2.0 * t * abs(np.cos(2.0 * np.pi * j / N)) for j in range(N // 2)])
    sp = np.sort(np.concatenate((-eps, eps)))
    return float(sp[: N // 2].sum())


def check_ks_continuum() -> dict:
    """P35：Lanczos 引擎 + 归一化标定 + 截断收敛 + 连续极限外推。

    **结构判据（进残差，全部应为机器精度或判据明确阈值）**
      ① matvec vs 稠密 H：同一个 H 的矩阵自由实现与 `ks_ed` 的稠密矩阵逐元素一致
      ② Lanczos vs 稠密 ED：两套独立本征求解器的最低两能级一致（N=8/10/12, Emax=2/3）
      ③ 自由极限（g=0、完备基）E0 = Σ 最低 N/2 个 ±2t|cos(k/2)|：钉住跃迁归一化
         与环上 JW 符号（PBC）
      ④ t=0 精确谱：H 只剩 (g²/2)ΣE²，gap 必须精确等于 g²/2 —— 钉住电场项归一化
      ⑤ Emax 截断：Δgap(3←2)/gap ≤ 5e-4（N=16,g=0.5 与 N=12,g=1，实测 8.8e-5 / 1.7e-7）
      ⑥ 表内三个点的现场复现（残差 ≤ 1e-6）：证明 GAP_SCAN 不是抄来的
    **物理结论（_magnitude，只报数不判定）**
      · 强耦合趋势 gap/(g²/2)（g=6 时 1.0016）
      · 连续极限 G(0) 与 1/√π 的偏差、拟合阶数敏感度（外推不确定度）
      · 两条判据（只防回归）：G(x) 随 x 单调下降；G(0) 落在 e/√π 的 [0.95,1.25] 窗口内
    """
    out = {}
    ked = _dense_module()

    # ① matvec vs 稠密 H
    worst = 0.0
    for N in (8, 12):
        H, masks, _ = ked.basis_and_hamiltonian(N, 1.0, 0.0, t=0.5, Emax=2)
        op = KogutSusskind1D(N, 1.0, t=0.5, Emax=2)
        rng = np.random.default_rng(5)
        v = rng.normal(size=op.dim)
        worst = max(worst, float(np.abs(op.matvec(v) - H @ v).max()))
    out["matvec vs 稠密 H（N=8/12）"] = worst

    # ② Lanczos vs 稠密 ED
    worst = 0.0
    for N in (8, 10, 12):
        for Emax in (2, 3):
            H, _, _ = ked.basis_and_hamiltonian(N, 1.0, 0.0, t=0.5, Emax=Emax)
            d = np.linalg.eigvalsh(H)
            g = gap_lanczos(N, 1.0, t=0.5, Emax=Emax, iters=200)
            worst = max(worst, abs(g["E0"] - d[0]), abs(g["E1"] - d[1]))
    out["Lanczos vs 稠密 ED（N=8/10/12, Emax=2/3）"] = worst

    # ③ 自由极限（完备基）
    worst = 0.0
    for N in (6, 8, 10):
        g = gap_lanczos(N, 0.0, t=0.5, Emax=N, iters=200)
        worst = max(worst, abs(g["E0"] - free_limit_energy(N, 0.5)))
    out["自由极限 E0 = Σ±2t|cos(k/2)|（PBC）"] = worst

    # ④ t=0：谱就是对角元，gap = g²/2
    g = gap_lanczos(8, 1.0, t=0.0, Emax=2, iters=40)
    out["t=0 精确谱 gap = g²/2"] = abs(g["gap"] - 0.5)

    # ⑤ Emax 截断
    tr = 0.0
    for (N, gg) in ((12, 1.0), (16, 0.5)):
        a = gap_lanczos(N, gg, t=0.5, Emax=2, iters=200)
        b = gap_lanczos(N, gg, t=0.5, Emax=3, iters=200)
        tr = max(tr, abs(b["gap"] - a["gap"]) / a["gap"])
    out["Emax 截断 Δgap(3←2)/gap（判据 ≤5e-4）"] = max(0.0, tr - 5e-4)
    out["Emax 截断 Δgap(3←2)/gap_magnitude"] = tr

    # ⑥ 表内三点现场复现
    worst = 0.0
    for (x, N) in ((1.0, 16), (0.7, 16), (0.5, 16)):
        tab = [v for (xx, nn, v) in GAP_SCAN if abs(xx - x) < 1e-9 and nn == N]
        if not tab:
            continue
        g = gap_lanczos(N, x, t=0.5, Emax=2, iters=200)
        worst = max(worst, abs(g["gap_over_e"] - tab[0]))
    out["表内三点现场复现"] = worst

    # 物理结论
    fit = fit_continuum_gap()
    out["连续极限 G(0) = gap/e_magnitude"] = fit["G0"]
    out["e/√π_magnitude"] = fit["target"]
    out["|G(0) − e/√π|/(e/√π)_magnitude"] = fit["dev_rel"]
    out["外推不确定度（拟合形式散布）_magnitude"] = fit["spread"]
    out["拟合最大偏差_magnitude"] = fit["max_dev"]
    out["格点伪影幂次 p_magnitude"] = fit["p"]
    for r in fit["rows"]:
        out["gap/e（x=%.2f, N→∞）_magnitude" % r["x"]] = r["G"]
    # 判据①：G(x) 随 x 单调下降（伪影随 x→0 收缩的定性证据）
    gs = [r["G"] for r in sorted(fit["rows"], key=lambda r: -r["x"])]
    out["G(x) 随 x 单调下降（判据）"] = float(max(0.0, max(b - a for a, b in zip(gs, gs[1:]))))
    # 判据②：外推值落在 [0.95, 1.25]·e/√π 窗口内（只防回归，不是精度声明；
    # 精度与负结果见 OPERATOR_STATUS.md §6terdecies）
    out["连续极限落在 e/√π 的 [0.95,1.25] 窗口（判据）"] = float(
        max(0.0, 0.95 * SQRT_PI_INV - fit["G0"], fit["G0"] - 1.25 * SQRT_PI_INV))
    # 强耦合标定（只报数）
    s = gap_lanczos(12, 6.0, t=0.5, Emax=3, iters=200)
    out["强耦合 gap/(g²/2)（g=6）_magnitude"] = s["gap"] / (0.5 * 36.0)
    return out
