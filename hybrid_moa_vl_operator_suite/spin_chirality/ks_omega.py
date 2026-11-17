# -*- coding: utf-8 -*-
"""动量分辨的光子谱 ω(k)：在 (mask, E) 平移协变基上按动量扇区分块，取每个扇区的最低激发。

为什么需要它
------------
P40 想用色散截距 ω² = m_γ² + Z k² 判"那个低能激发是不是光子"，失败在**基**上：
`ks_dispersion.py` 的 `dispersion_point(..., j)` 里 j 只换投影权重，五个 j 给出同一个 ω
（N=12、x=0.4、φ=0 时都是 0.371983），平移算符在 g≠0 也与 H 不对易（‖[H,T]‖ = 0.16/0.64/4.0）。
P41 把基重写成 (mask, E)、让 T 成纯置换以后，`[H,T] = 0` 精确，按 k 分扇区才第一次可做。

本模块做三件事
--------------
① `dispersion(N, g)`：在每个动量扇区里求最低两个本征值，ω(k) = E₁(k) − E₀（k = 0 的扇区里
   最低态就是基态，取第二个）。不物化稠密矩阵：H 存成 (row, col, val) 三元组，matvec 用
   `np.bincount` 做；扇区基是轨道上的 Fourier 组合，Q†HQ 也是同一套三元组，于是纯 numpy、
   N 能到 18（dim = 1.4e5）。
② `OMEGA_SCAN`：N = 16 上 6 个耦合的实测表（由根目录 `ks_omega_scan.py` 重算），
   以及从表里读出的两个判据——质量的连续极限与色散斜率。
③ `check_omega()`：P42 的判据，残差表。

模型与约定（与 `ks_basis.py` 逐项一致）
--------------------------------------
    H = (g²/2) Σ_x E_x²  −  t Σ_x (χ†_x U_x χ_{x+1} + h.c.)  +  m Σ_x (−1)^x n_x
    x = g/√(2t) 是无量纲耦合（t = 0.5 时 x = g）；连续极限是 x → 0。
    光子质量在连续极限里是 m_γ/e = 1/√π = 0.5641895，
    色散是 ω² = m_γ² + k²，即在格点单位下 ω_lat² = (x²/π) + k² + O(x²)。

动量网格（容易错）
------------------
平移的单位胞是 2 格（交错费米子），所以 T 的阶是 N/2，动量是 k = 2πj/N（j = 0…N/2−1），
不是 2πj/(N/2)。谱在 k → π − k 下相同（BZ 折叠），本模块统一折到 [0, π/2]；
k 与 k+π 共用一个 T 本征值，所以能区只有 π。

实测（N = 16、Emax = 2、t = 0.5）
--------------------------------
  x = 1.0 : ω = 0.7846, 0.8605(0.393), 1.0377(0.785), 1.2317(1.178), 1.3806(1.571)
  x = 0.2 : ω = 0.1194, 0.4054,          0.7527,        1.0164,        1.1134
  同一个 k 上 N = 8/12/16 一致到 7e-4（k = π/2 处 1.3806 vs 1.3807），有限尺寸误差很小；
  E 截断 Emax = 2 → 3 在 N = 8/12 上谱与 gap 都不动（0.783991 / 0.784626，逐位相同）。

两个判据（都在 `check_omega` 里，数字见 `OMEGA_SCAN`）
----------------------------------------------------
  ① 质量：G(x) ≡ ω(0)/x 随 x → 0 单调下降，x = 1/0.7/0.5/0.4/0.3/0.2 给
     0.7846 / 0.7163 / 0.6725 / 0.6497 / 0.6248 / 0.5969；x ≤ 0.5 的三种外推
     0.5479 / 0.5885 / 0.5338（均值 0.5567、半宽 0.027），对 1/√π = 0.56419 差 1.3%，
     而 1/√π 落在三形式张成的区间里。
  ② 斜率：Z_loc ≡ (ω(k₁)² − ω(0)²)/k₁²（k₁ = 2π/N = 0.3927）单调趋 1：
     0.810 / 0.891 / 0.934 / 0.950 / 0.963 / 0.973，x ≤ 0.5 的三种外推
     1.0011 / 0.9802 / 0.9842（均值 0.9885），Z(0) − 1 = −0.012。
  ③ 二维（`GAP_2D`，7 个 x × 6 个 N = 42 点）：把有限尺寸与格点间距分开以后，
     六种联合拟合 `G(x,N) = A0 + a·x + [x²] + [1/N² 或 e^{−xN/√π}]` 给
     A0 = 0.5539 / 0.5450 / 0.5632 / 0.5543 / 0.5631 / 0.5750（均值 0.5591、散布 5.3%）。
     其中 x² 项**不成立**：它在两种有限尺寸处理下的系数是 −0.0329 ± 0.0086 与
     +0.0318 ± 0.0058——同一个物理修正项不会因为有限尺寸换个写法就翻号，可见它在替
     模型失配背锅。删掉 x² 之后只剩三种（1/N²、1/N、e^{−m_γN}），A0 = 0.55389 ± 0.00164 /
     0.56322 ± 0.00246 / 0.56310 ± 0.00143：**散布 1.67% < 2%（判据 ② 过）**，
     均值 0.56007，对 1/√π 差 0.73%。
  这就是 P40 没测出来的那条色散：Z 不是 0，也不是 1.1，而是从 0.82 单调趋于 1。

边界（这三条别忘）
------------------
  · 可及动量 k₁ = 2π/N 并不"小"：x = 0.2 时 k₁/m_γ = 0.393/0.120 = 3.3，所以 Z_loc 是
    "最小可及动量上的平方斜率"，不是 k → 0 的严格斜率；外推给的是它的 x → 0 极限。
  · 判据 ② 的 2% 依赖"x² 项被数据否掉"这一步模型选择；如果把 x² 也留着，六种形式的散布是
    5.3%——两种读数都在 `model_selection()` 里，不要只引对自己有利的那个。
  · 大 k 处的最轻态会与两光子阈值重叠（y = k/x ≳ 2 时实测比 √(x²/π + k²) 低 4–25%），
    那里"是不是光子"这句话没有意义；本项只用 k ≲ 0.4 与 k → 0 的趋势。
  · N ≤ 18、Emax = 2、m = 0；m ≠ 0 与更大 N 没做（要更小 x 得 N ≳ 3/(0.564 x)，x = 0.15 要
    N ≳ 35，那个尺寸必须换 DMRG，见 OPERATOR_STATUS.md 的同名小节）。
"""
from __future__ import annotations

import numpy as np

from . import ks_basis as _kb

__all__ = ["dispersion", "omega_scan", "OMEGA_SCAN", "mass_over_e", "z_local",
           "extrapolate", "check_omega"]

SQRT_PI_INV = float(1.0 / np.sqrt(np.pi))          # 0.5641895…


# --------------------------------------------------------------------------
# H 的三元组（与 ks_basis.build_h 逐项一致，只是不物化 dim×dim）
# --------------------------------------------------------------------------
def _h_triples(N: int, g: float, m: float = 0.0, t: float = 0.5, Emax: int = 2):
    states, index = _kb.enumerate_basis(N, Emax)
    dim = len(states)
    diag = np.zeros(dim)
    for i, (mask, E) in enumerate(states):
        d = 0.5 * g * g * float(sum(e * e for e in E))
        d += m * sum((1 if x % 2 == 0 else -1) * ((mask >> x) & 1) for x in range(N))
        diag[i] = d
    rows, cols, vals = [], [], []
    for i, (mask, E) in enumerate(states):
        for x in range(N):
            y = (x + 1) % N
            for a, b in ((x, y), (y, x)):          # 两个方向都写，越界就整对丢掉
                if ((mask >> b) & 1) and not ((mask >> a) & 1):
                    nm = mask ^ (1 << a) ^ (1 << b)
                    nE = list(E)
                    nE[x] += 1 if a == x else -1    # 只动链接 x
                    j = index.get((nm, tuple(nE)))
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        vals.append(-t)
    return (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
            np.array(vals, dtype=float), diag, states, index)


def _binc(rows, vals, n):
    """bincount 的复数版：numpy 的 bincount 只吃实数权重。"""
    re = np.bincount(rows, weights=vals.real, minlength=n)
    im = np.bincount(rows, weights=vals.imag, minlength=n)
    return re + 1j * im


def _matvec_h(u, rows, cols, vals, diag):
    return _binc(rows, vals * u[cols], diag.size) + diag * u


def _orbit_fourier(N: int, states, index):
    """轨道上的 Fourier 组合：返回 Q 的三元组与每列的 (m, L) 标签。

    Q 的列就是 T 的本征矢（m/L 已约分），nnz 只有 Σ L = dim·(平均轨道长)。
    """
    perm = _kb._permutation(N, states, index, 2)
    dim = len(states)
    seen = [False] * dim
    qr, qc, qv, meta = [], [], [], []
    ncol = 0
    for start in range(dim):
        if seen[start]:
            continue
        orbit, i = [], start
        while not seen[i]:
            seen[i] = True
            orbit.append(i)
            i = perm[i]
        L = len(orbit)
        idx = np.array(orbit)
        for mm in range(L):
            gg = int(np.gcd(mm, L))
            qr.append(idx)
            qc.append(np.full(L, ncol))
            qv.append(np.exp(-2j * np.pi * mm * np.arange(L) / L) / np.sqrt(L))
            meta.append((mm // gg, L // gg))
            ncol += 1
    return (np.concatenate(qr), np.concatenate(qc), np.concatenate(qv), meta, ncol)


def _lanczos(matvec, dim: int, k: int = 2, iters: int = 90, seed: int = 3):
    """复数 Lanczos，两遍全重正交，返回最低 k 个 Ritz 值。"""
    rng = np.random.default_rng(seed)
    V = np.zeros((iters, dim), dtype=complex)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v /= np.linalg.norm(v)
    V[0] = v
    al = np.zeros(iters)
    be = np.zeros(iters)
    used = 0
    for j in range(iters):
        w = np.asarray(matvec(V[j]), dtype=complex)
        al[j] = float(np.real(np.vdot(V[j], w)))
        if j > 0:
            w -= be[j - 1] * V[j - 1]
        w -= al[j] * V[j]
        for _ in range(2):
            w -= (V[: j + 1].conj() @ w) @ V[: j + 1]
        used = j + 1
        be[j] = float(np.linalg.norm(w))
        if be[j] < 1e-12 or j == iters - 1:
            break
        V[j + 1] = w / be[j]
    T = np.diag(al[:used])
    if used > 1:
        T += np.diag(be[: used - 1], 1) + np.diag(be[: used - 1], -1)
    return np.sort(np.linalg.eigvalsh(T))[:k]


def dispersion(N: int, g: float, m: float = 0.0, t: float = 0.5, Emax: int = 2,
               iters: int = 90, kmax: float | None = None, nkeep: int = 2):
    """按动量扇区求 ω(k)。

    返回 `(ks, omegas, nsec)`：`ks` 折到 [0, π/2] 的动量（k = 2πj/N），`omegas[i]` 是该扇区
    最低激发相对全局基态的能量；`nsec` 是实际解过的扇区数（`kmax` 用来只解一部分）。
    """
    rows, cols, vals, diag, states, index = _h_triples(N, g, m, t, Emax)
    qr, qc, qv, meta, ncol = _orbit_fourier(N, states, index)
    groups = {}
    for c, lab in enumerate(meta):
        groups.setdefault(lab, []).append(c)

    def mv_full(x):
        u = _binc(qr, qv * x[qc], diag.size)          # Q x
        v = _matvec_h(u, rows, cols, vals, diag)      # H Q x
        return _binc(qc, np.conj(qv) * v[qr], ncol)   # Q† H Q x

    picked, e0 = {}, None
    for lab, colsel in groups.items():
        k = np.pi * lab[0] / lab[1]
        if kmax is not None and k > kmax + 1e-9:
            continue
        colsel = np.asarray(colsel)

        def mv(x, colsel=colsel):
            full = np.zeros(ncol, dtype=complex)
            full[colsel] = x
            return mv_full(full)[colsel]

        ew = _lanczos(mv, len(colsel), k=max(nkeep, 2 if k < 1e-9 else 1), iters=iters)
        picked[lab] = ew
        e0 = ew[0] if e0 is None else min(e0, ew[0])
    out = {}
    for lab, ew in picked.items():
        k = np.pi * lab[0] / lab[1]
        w = (ew[1] if (k < 1e-9 and len(ew) > 1) else ew[0]) - e0
        kk = round(min(k, np.pi - k), 9)
        out.setdefault(kk, w)
    ks = np.array(sorted(out))
    return ks, np.array([out[k] for k in ks]), len(picked)


# --------------------------------------------------------------------------
# 记录的扫描表（N = 16；根目录 ks_omega_scan.py 重算）
# --------------------------------------------------------------------------
OMEGA_SCAN = {
    1.0: [0.7846470165687922, 0.8605235907643434, 1.0377038629518434,
          1.2317379982995558, 1.3806477660365322],
    0.7: [0.5014290343498371, 0.6236160799564425, 0.8709238178494769,
          1.1098563062017326, 1.2585922539006358],
    0.5: [0.3362514847197122, 0.5069669954055547, 0.8024725734274867,
          1.0632389094258077, 1.1948049179490225],
    0.4: [0.2598720064503013, 0.4625992953921392, 0.7793992314371070,
          1.0459958513598897, 1.1641535000604408],
    0.3: [0.18745239756408516, 0.4285259671424013, 0.7629809964420007,
          1.0305666941190710, 1.1358194986507462],
    0.2: [0.11937801125634984, 0.4053596889641433, 0.7527080894393778,
          1.0164463520266183, 1.1133850175493079],
}
SCAN_N = 16
# 折叠到 [0, π/2] 之后的动量网格：k = 2πj/16 与 π − k 合并
SCAN_K = [0.0, float(2 * np.pi / 16), float(np.pi / 2),
          float(2 * np.pi / 16 * 3), float(np.pi / 2)]


def omega_scan(N: int = SCAN_N, xs=(1.0, 0.7, 0.5, 0.4, 0.3, 0.2), iters: int = 90,
               verbose: bool = False) -> dict:
    """把 `OMEGA_SCAN` 重算一遍（N = 16 全扇区，纯 numpy）。返回 {x: [ω(k)…]}。"""
    out = {}
    for x in xs:
        ks, ws, nsec = dispersion(N, x, iters=iters)
        out[float(x)] = [float(w) for w in ws]
        if verbose:
            print(f"  x={x:.2f} 扇区 {nsec}  " +
                  " ".join(f"{k:.3f}:{w:.6f}" for k, w in zip(ks, ws)), flush=True)
    return out


def mass_over_e(table: dict | None = None) -> dict:
    """G(x) = ω(0)/x：格点单位下的质量除以 e（x = e·a）。"""
    table = OMEGA_SCAN if table is None else table
    return {x: rows[0] / x for x, rows in sorted(table.items())}


def z_local(table: dict | None = None, N: int = SCAN_N) -> dict:
    """Z_loc(x) = (ω(k₁)² − ω(0)²)/k₁²，k₁ = 2π/N（最小可及动量）。"""
    table = OMEGA_SCAN if table is None else table
    k1 = 2 * np.pi / N
    return {x: (rows[1] ** 2 - rows[0] ** 2) / k1 ** 2 for x, rows in sorted(table.items())}


def extrapolate(table: dict | None = None, key: str = "mass", xs_max: float = 0.5):
    """x → 0 的三种外推：线性 x、线性 x²、二次 x。返回 (值列表, 均值, 半宽)。"""
    table = OMEGA_SCAN if table is None else table
    ys = mass_over_e(table) if key == "mass" else z_local(table)
    xs = np.array([x for x in sorted(ys) if x <= xs_max + 1e-12])
    yy = np.array([ys[x] for x in xs])
    vals = []
    for A in (np.vstack([xs, np.ones_like(xs)]).T,
              np.vstack([xs ** 2, np.ones_like(xs)]).T,
              np.vstack([xs ** 2, xs, np.ones_like(xs)]).T):
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        vals.append(float(coef[-1]))
    vals = np.array(vals)
    return vals, float(vals.mean()), float(vals.max() - vals.min())


# --------------------------------------------------------------------------
# (x, N) 二维表：有限尺寸与格点间距分开看
# --------------------------------------------------------------------------
# gap(x, N)（t = 0.5、Emax = 2、m = 0；k = 0 扇区的 E₁ − E₀）。
# 复现：根目录 `python ks_omega_scan.py --2d`（N = 18 那一列约 100 s）。
GAP_2D_NS = (8, 10, 12, 14, 16, 18)
GAP_2D = {
    1.0: [0.783990834490, 0.784529642449, 0.784626211005,
          0.784643785166, 0.784647016569, 0.784647615160],
    0.7: [0.498131031264, 0.500374595463, 0.501106231816,
          0.501348100664, 0.501429034350, 0.501456365827],
    0.5: [0.329715573263, 0.333362224779, 0.335074155963,
          0.335874659834, 0.336251484720, 0.336430329323],
    0.4: [0.252342878935, 0.255998169731, 0.258078900706,
          0.259233483936, 0.259872006450, 0.260226427336],
    0.3: [0.180687562061, 0.183382207511, 0.185312487950,
          0.186606049158, 0.187452397564, 0.188001940743],
    0.25: [0.147387943874, 0.149242336898, 0.150798638293,
           0.151968125108, 0.152809922759, 0.153404934614],
    0.2: [0.115959764000, 0.116819755045, 0.117803277492,
          0.118672792334, 0.119378011256, 0.119928512357],
}


def gap_2d(N: int, x: float, iters: int = 90) -> float:
    """只解 k = 0 扇区的 gap（比 `dispersion` 快 8 倍：一个扇区而不是八个）。"""
    _k, w, _n = dispersion(N, x, iters=iters, kmax=0.0)
    return float(w[0])


_JOINT_FORMS = {
    "1+x+1/N²": lambda x, n: [1.0, x, 1.0 / n ** 2],
    "1+x+x²+1/N²": lambda x, n: [1.0, x, x ** 2, 1.0 / n ** 2],
    "1+x+1/N": lambda x, n: [1.0, x, 1.0 / n],
    "1+x+x²+1/N": lambda x, n: [1.0, x, x ** 2, 1.0 / n],
    "1+x+exp(−xN/√π)": lambda x, n: [1.0, x, np.exp(-SQRT_PI_INV * x * n)],
    "1+x+x²+exp(−xN/√π)": lambda x, n: [1.0, x, x ** 2, np.exp(-SQRT_PI_INV * x * n)],
}


def joint_fit(table: dict | None = None, ns=GAP_2D_NS) -> dict:
    """42 个点一起拟合 G(x,N) = A0 + a·x + [x²] + [有限尺寸项]，返回每种形式的 A0 与误差。

    有限尺寸项给两种（1/N²、e^{−m_γ N}）、x 项给线性与带 x² 两种 ⇒ 六种形式；
    A0 的散布就是"外推模型不确定度"的直接读数。
    """
    table = GAP_2D if table is None else table
    rows, y = [], []
    for x, vals in table.items():
        for n, g in zip(ns, vals):
            rows.append((x, float(n)))
            y.append(float(g) / x)          # G(x, N) = gap / x
    y = np.array(y)
    out = {}
    for name, basis in _JOINT_FORMS.items():
        A = np.array([basis(x, n) for x, n in rows])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = A @ coef - y
        dof = max(len(y) - A.shape[1], 1)
        cov = np.linalg.pinv(A.T @ A)
        cov = cov * float(res @ res) / dof
        err = np.sqrt(np.diag(cov))
        out[name] = {"A0": float(coef[0]), "err": float(err[0]),
                     "coef": coef.copy(), "cerr": err.copy(),
                     "maxres": float(np.abs(res).max()),
                     "rms": float(np.sqrt((res ** 2).mean()))}
    return out


# 首选模型族：x² 项被剔除之后剩下的三种（唯一的歧义只剩有限尺寸怎么写）
_CORE_FORMS = ("1+x+1/N²", "1+x+1/N", "1+x+exp(−xN/√π)")


def model_selection(table: dict | None = None, ns=GAP_2D_NS) -> dict:
    """为什么可以把 x² 项删掉：它在两种有限尺寸处理下**变号**。

    · 1+x+x²+1/N²：x² 系数 −0.0329 ± 0.0086（t = 3.8）
    · 1+x+x²+exp ：x² 系数 +0.0318 ± 0.0058（t = 5.5）

    同一个"物理"修正项不会因为有限尺寸项换个写法就翻符号 ⇒ 它是在替模型失配背锅。
    删掉之后剩三种（有限尺寸 1/N²、1/N、e^{−m_γN}），A0 的散布就是判据 ② 的读数。
    """
    jf = joint_fit(table, ns)
    c_n2 = jf["1+x+x²+1/N²"]
    c_ex = jf["1+x+x²+exp(−xN/√π)"]
    flip = (c_n2["coef"][2] * c_ex["coef"][2]) < 0.0
    a0 = np.array([jf[k]["A0"] for k in _CORE_FORMS])
    err = np.array([jf[k]["err"] for k in _CORE_FORMS])
    return {
        "A0": a0, "err": err,
        "spread": float(a0.max() - a0.min()),
        "spread_rel": float((a0.max() - a0.min()) / a0.mean()),
        "mean": float(a0.mean()),
        "dev_vs_sqrtpi": float(abs(a0.mean() - SQRT_PI_INV) / SQRT_PI_INV),
        "x2_sign_flips": bool(flip),
        "x2_n2": (float(c_n2["coef"][2]), float(c_n2["cerr"][2])),
        "x2_exp": (float(c_ex["coef"][2]), float(c_ex["cerr"][2])),
        "spread_all6": float(np.ptp([v["A0"] for v in jf.values()])),
        "joint": jf,
    }


# --------------------------------------------------------------------------
# P42 的验收
# --------------------------------------------------------------------------
def _dense_reference(N: int, g: float, m: float = 0.0, t: float = 0.5, Emax: int = 2):
    """稠密 ED 的扇区最低谱（只给 N ≤ 12 用）：用来验 Lanczos 与三元组 matvec。"""
    H, states, index = _kb.build_h(N, g, m, t, Emax)
    labels, U = _kb.momentum_sectors(N, states, index)
    codes = _kb._label_codes([tuple(x) for x in labels])
    w = np.linalg.eigvalsh(H)
    e0 = w[0]
    out = {}
    for code in sorted(set(codes.tolist())):
        sel = np.where(codes == code)[0]
        Uk = U[:, sel]
        ew = np.linalg.eigvalsh(Uk.conj().T @ H @ Uk)
        lab = labels[sel[0]]
        k = np.pi * lab[0] / lab[1]
        ww = (ew[1] if (k < 1e-9 and len(ew) > 1) else ew[0]) - e0
        out.setdefault(round(min(k, np.pi - k), 9), ww)
    ks = np.array(sorted(out))
    return ks, np.array([out[k] for k in ks]), H, states, index, U, labels


def _field_weights(N: int, g: float, m: float = 0.0, t: float = 0.5, Emax: int = 2):
    """<n|E_x|0> 的动量结构：返回 (|k| 上的权重表, 求和规则残差)。"""
    H, states, index = _kb.build_h(N, g, m, t, Emax)
    w, V = np.linalg.eigh(H)
    v0 = V[:, 0]
    Eflat = np.array([[E[x] for x in range(N)] for _mask, E in states], dtype=float)
    G = V.T @ (Eflat * v0[:, None])            # (dim, N)
    amp = np.abs(np.fft.fft(G, axis=1) / np.sqrt(N)) ** 2
    tot = amp.sum(axis=1)
    rule = abs(tot.sum() - float((Eflat[:, 0] ** 2 * v0 ** 2).sum() * N))
    return w, tot, rule


def check_omega() -> dict:
    """P42：动量分辨的光子色散 ω(k)（`ks_omega.py`）。

    判据全部是残差（< 1e-5 才算过）；只作读数的以 `_magnitude` 结尾。
    """
    out = {}
    # ① Lanczos + 三元组 vs 稠密 ED（同一个 H，两条独立实现）
    err = 0.0
    for N in (8, 12):
        kd, wd, *_ = _dense_reference(N, 1.0)
        kn, wn, _ = dispersion(N, 1.0)
        m = min(len(kd), len(kn))
        err = max(err, float(np.abs(np.array(wd[:m]) - np.array(wn[:m])).max()))
    out["扇区谱 vs 稠密 ED（N=8/12, g=1）"] = err

    # ② 有限尺寸：同一个 k = π/2 在两个尺寸上（N=8 与 N=12）
    k8, w8, _ = dispersion(8, 1.0)
    k12, w12, _ = dispersion(12, 1.0)
    i8 = int(np.argmin(np.abs(k8 - np.pi / 2)))
    i12 = int(np.argmin(np.abs(k12 - np.pi / 2)))
    fin = float(abs(w8[i8] - w12[i12]) / w12[i12])
    out["有限尺寸 ω(π/2) 偏差 > 2e-3（0/1）"] = 0.0 if fin < 2e-3 else 1.0
    out["有限尺寸 ω(π/2)：N=8 vs N=12（相对）_magnitude"] = fin

    # ③ BZ 折叠：E(k) = E(π − k)（N = 12 的 k 与 π − k 是两个不同扇区）
    kd, wd, H, states, index, U, labels = _dense_reference(12, 1.0)
    codes = _kb._label_codes([tuple(x) for x in labels])
    raw = {}
    for code in sorted(set(codes.tolist())):
        sel = np.where(codes == code)[0]
        lab = labels[sel[0]]
        raw[round(np.pi * lab[0] / lab[1], 9)] = float(
            np.linalg.eigvalsh(U[:, sel].conj().T @ H @ U[:, sel])[0])
    e0 = min(raw.values())
    pair_err = 0.0
    for k, val in raw.items():
        if k <= np.pi / 2 + 1e-9 and (round(np.pi - k, 9) in raw):
            pair_err = max(pair_err, abs((val - e0) - (raw[round(np.pi - k, 9)] - e0)))
    out["BZ 折叠 E(k)=E(π−k)（N=12）"] = pair_err

    # ④ 电场求和规则：Σ_n |<n|E_x|0>|² = <E_x²>
    w, tot, rule = _field_weights(8, 1.0)
    out["电场求和规则 Σ_n|<n|E_x|0>|² = <E_x²>"] = rule

    # ⑤ k=0 的最低激发必须是"最像光子"的那个（E 权重最大）
    k0w = tot[1]
    others = tot[2:]
    out["k=0 最低激发的 E 权重最大（0/1）"] = 0.0 if k0w >= others.max() else 1.0
    out["k=0 最低激发的 E 权重_magnitude"] = float(k0w / tot.sum())

    # ⑥ 与 P35 的 gap 在 N=8 上比一眼（只作读数：q=0 切片固定 E_{N−1}=0，不是不变子空间，
    #    它的 Ritz 值是上界，N=8 上低 10%，N 增大时两者收敛）
    try:
        from . import ks_lanczos as _lan
        gap35 = float(_lan.gap_lanczos(8, 1.0)["gap"])
        out["ω(0) vs P35 的 gap 之差（N=8, x=1）_magnitude"] = abs(w8[0] - gap35) / gap35
    except Exception:
        pass

    # ⑦ 表内两行现场复现（N = 16）
    rep = 0.0
    for x in (1.0, 0.2):
        _k, ws, _n = dispersion(SCAN_N, x)
        row = OMEGA_SCAN[x]
        n = min(len(ws), len(row))
        rep = max(rep, float(np.abs(ws[:n] - np.array(row[:n])).max()))
    out["OMEGA_SCAN 表复现（N=16，x=1.0/0.2）"] = rep

    # ⑧ 斜率随 x → 0 单调趋 1
    zs = z_local()
    xs = sorted(zs)
    mono = all(zs[xs[i]] >= zs[xs[i + 1]] - 1e-12 for i in range(len(xs) - 1))
    out["Z_loc 随 x 减小单调上升（0/1）"] = 0.0 if mono else 1.0
    out["Z_loc(x=0.2) 距 1 超过 10%（0/1）"] = 0.0 if abs(zs[0.2] - 1.0) < 0.10 else 1.0
    out["Z_loc(x=0.2)_magnitude"] = float(zs[0.2])
    zv, zmean, zspread = extrapolate(key="z")
    out["Z(0) 三形式外推距 1 超过 0.05（0/1）"] = 0.0 if abs(zmean - 1.0) < 0.05 else 1.0
    out["Z(0) 外推散布_magnitude"] = zspread
    out["Z(0) 三形式_magnitude"] = float(zv[0])
    out["Z(0) 三形式2_magnitude"] = float(zv[1])
    out["Z(0) 三形式3_magnitude"] = float(zv[2])

    # ⑨ 质量外推到 1/√π（判据：相对偏差 < 5%，且 1/√π 落在三形式张成的区间里）
    gv, gmean, gspread = extrapolate(key="mass")
    dev = abs(gmean - SQRT_PI_INV) / SQRT_PI_INV
    inside = (gv.min() <= SQRT_PI_INV <= gv.max())
    out["G(0) 三形式外推 vs 1/√π 相对偏差 > 5%（0/1）"] = 0.0 if dev < 0.05 else 1.0
    out["1/√π 不在三形式区间里（0/1）"] = 0.0 if inside else 1.0
    out["G(0) 外推散布_magnitude"] = gspread
    out["G(0) 相对偏差_magnitude"] = dev
    out["G(0)_magnitude"] = gmean
    out["1/√π_magnitude"] = SQRT_PI_INV

    # ⑩ (x, N) 二维：表复现 + 六种联合拟合
    rep2 = max(abs(gap_2d(14, 0.5) - GAP_2D[0.5][GAP_2D_NS.index(14)]),
               abs(gap_2d(12, 0.2) - GAP_2D[0.2][GAP_2D_NS.index(12)]))
    out["GAP_2D 表复现（x=0.5/N=14、x=0.2/N=12）"] = float(rep2)
    ms = model_selection()
    a0 = ms["A0"]
    out["x² 项在两种有限尺寸处理下变号（0/1）"] = 0.0 if ms["x2_sign_flips"] else 1.0
    out["首选族三形式散布超过 2%（0/1）"] = 0.0 if ms["spread_rel"] < 0.02 else 1.0
    out["首选族均值距 1/√π 超过 5%（0/1）"] = 0.0 if ms["dev_vs_sqrtpi"] < 0.05 else 1.0
    out["首选族三形式散布（相对）_magnitude"] = ms["spread_rel"]
    out["首选族均值_magnitude"] = ms["mean"]
    out["首选族均值 vs 1/√π（相对）_magnitude"] = ms["dev_vs_sqrtpi"]
    out["x² 系数（1/N² 处理）_magnitude"] = ms["x2_n2"][0]
    out["x² 系数（exp 处理）_magnitude"] = ms["x2_exp"][0]
    out["六种形式全收的散布_magnitude"] = ms["spread_all6"]
    for k, nm in zip(("1/N²", "1/N", "exp"), _CORE_FORMS):
        out[f"  A0[{k}]_magnitude"] = float(ms["joint"][nm]["A0"])
        out[f"  A0[{k}] 误差_magnitude"] = float(ms["joint"][nm]["err"])
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check_omega(), ensure_ascii=False, indent=2, default=float))
