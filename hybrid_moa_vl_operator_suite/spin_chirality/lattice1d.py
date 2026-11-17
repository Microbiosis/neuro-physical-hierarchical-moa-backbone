# -*- coding: utf-8 -*-
"""1+1D Wilson–Dirac 格点：色散关系、费米子倍增、畴壁零模（P22 / P23 的底层模块）。

这是把算符从「0+1D 内部空间」推进到「1+1D 有空间」的第一步：给内部演化加一条真正的
空间方向（最近邻跃迁），于是第一次出现 **色散关系**、**费米子倍增**、**拓扑零模**。

约定
----
2 分量旋量，``γ⁰ = σ_x``、``γ¹ = iσ_y`` ⇒ ``α = γ⁰γ¹ = σ_z``，1+1D 里 ``γ5 = α = σ_z``。

    H = Σ_x [ (i/2)(ψ†_{x+1} σ_z ψ_x − ψ†_x σ_z ψ_{x+1})
              + m_x ψ†_x σ_x ψ_x
              + (r/2)(2ψ†_x σ_x ψ_x − ψ†_{x+1} σ_x ψ_x − ψ†_x σ_x ψ_{x+1}) ]

常数 m、周期边界：``H(k) = σ_z sin k + σ_x (m + r(1 − cos k))``，
``E(k) = ±√( sin²k + (m + r(1−cos k))² )``。
``r = 0``（naive）在 ``k = 0`` 与 ``k = π`` 各有一个同质量狄拉克点 ⇒ **倍增**；
``r = 1``（Wilson）把 ``k = π`` 那支的质量抬到 ``m + 2r`` ⇒ 倍增被消掉。

只依赖 numpy。
"""
from __future__ import annotations

import numpy as np

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def build(N: int, m, r: float = 1.0, periodic: bool = True, theta=None, V=None) -> np.ndarray:
    """2N×2N 厄米哈密顿量。

    `m`   : 标量或长度 N 的数组（空间变化的 ``m_x``）
    `theta`: U(1) 规范链接的相位，每个键一个（周期边界长度 N，开边界 N−1）；缺省为 0
    `V`   : 每格点的标量势 ``V_x``（σ_0 项），缺省为 0
    """
    if N < 2:
        raise ValueError("N 至少为 2")
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    mx = np.full(N, float(m)) if np.isscalar(m) else np.asarray(m, dtype=float)
    if mx.shape != (N,):
        raise ValueError("m 必须是标量或长度 N 的数组")
    bonds = [(x, x + 1) for x in range(N - 1)]
    if periodic:
        bonds.append((N - 1, 0))
    if theta is not None:
        th = np.asarray(theta, dtype=float)
        if th.shape != (len(bonds),):
            raise ValueError(f"theta 需要 {len(bonds)} 个相位（每键一个）")
    else:
        th = np.zeros(len(bonds))
    for idx, (x, y) in enumerate(bonds):
        up_ = np.exp(1j * th[idx])          # Peierls 相位：规范协变跃迁
        dn_ = np.conj(up_)
        H[2 * y:2 * y + 2, 2 * x:2 * x + 2] += 0.5j * SZ * up_
        H[2 * x:2 * x + 2, 2 * y:2 * y + 2] += -0.5j * SZ * dn_
        H[2 * y:2 * y + 2, 2 * x:2 * x + 2] += -0.5 * r * SX * up_
        H[2 * x:2 * x + 2, 2 * y:2 * y + 2] += -0.5 * r * SX * dn_
    for x in range(N):
        H[2 * x:2 * x + 2, 2 * x:2 * x + 2] += (mx[x] + r) * SX
    if V is not None:
        vv = np.asarray(V, dtype=float)
        if vv.shape != (N,):
            raise ValueError("V 需要长度 N 的数组")
        for x in range(N):
            H[2 * x:2 * x + 2, 2 * x:2 * x + 2] += vv[x] * np.eye(2)
    return H


def analytic_spectrum(N: int, m: float, r: float = 1.0) -> np.ndarray:
    """周期边界、常数质量下的精确谱（用来和数值谱对拍）。"""
    k = 2.0 * np.pi * np.arange(N) / N
    E = np.sqrt(np.sin(k) ** 2 + (m + r * (1.0 - np.cos(k))) ** 2)
    return np.sort(np.concatenate([E, -E]))


def light_species(N: int, m: float, r: float, tol: float = 1e-9):
    """色散曲线上「与全局最小值等高的局域极小」的个数 = 狄拉克点个数。

    r=0：k=0 与 k=π 两个极小、同高 ⇒ 2（倍增）；r=1：只剩 k=0 ⇒ 1。
    """
    k = 2.0 * np.pi * np.arange(N) / N
    E = np.sqrt(np.sin(k) ** 2 + (m + r * (1.0 - np.cos(k))) ** 2)
    emin = E.min()
    is_min = (E <= np.roll(E, 1) + 1e-12) & (E <= np.roll(E, -1) + 1e-12)
    return int(np.sum(is_min & (np.abs(E - emin) < tol)))


def wall_profile(N: int, m0: float = 1.0, eps: float = 1.5) -> np.ndarray:
    """环上的畴壁对：m 在 x=0 与 x=N/2 各变一次号（kink + antikink）。"""
    x = np.arange(N, dtype=float)
    return m0 * np.tanh(np.sin(2.0 * np.pi * x / N) / eps)


def zero_modes(N: int, m0: float = 1.0, eps: float = 1.5, r: float = 1.0,
               tol_frac: float = 1e-6):
    """畴壁环上的零模。返回 (零模能量数组, 每个零模的格点密度 (n, N), 体隙)。"""
    H = build(N, wall_profile(N, m0, eps), r, periodic=True)
    w, v = np.linalg.eigh(H)
    order = np.argsort(np.abs(w))
    ws = np.abs(w[order])
    tol = tol_frac * float(np.abs(w).max())
    nz = int(np.sum(ws < tol))
    dens = np.array([np.abs(v[order[j], 0::2]) ** 2 + np.abs(v[order[j], 1::2]) ** 2
                     for j in range(nz)])
    bulk = float(ws[nz]) if nz < len(ws) else 0.0
    return w[order[:nz]], dens, bulk


def check_dispersion(N: int = 64, m: float = 0.3) -> dict:
    """P22：色散关系与费米子倍增（用 Wilson 项消倍增）。"""
    out = {}
    for r in (0.0, 1.0):
        num = np.linalg.eigvalsh(build(N, m, r, periodic=True))
        out[f"色散 E(k) vs 解析 (r={r:g})"] = float(
            np.abs(num - analytic_spectrum(N, m, r)).max())
    out["狄拉克点数 (r=0) = 2（倍增）"] = abs(light_species(N, m, 0.0) - 2)
    out["狄拉克点数 (r=1) = 1（Wilson 消倍增）"] = abs(light_species(N, m, 1.0) - 1)
    return out


def check_domain_wall(N: int = 160, m0: float = 1.0, eps: float = 1.5) -> dict:
    """P23：畴壁零模——Wilson 下一个壁一个；naive 下因倍增翻倍。"""
    e1, d1, bulk1 = zero_modes(N, m0, eps, r=1.0)
    e0, _, _ = zero_modes(N, m0, eps, r=0.0)
    out = {
        "零模数 (r=1) = 2": abs(len(e1) - 2),
        "零模数 (r=0) = 4（倍增）": abs(len(e0) - 4),
        "零模能量 / 体隙": float(np.max(np.abs(e1)) / max(bulk1, 1e-12)),
    }
    ratio = float(d1.max() / max(d1.mean(), 1e-12))
    out["定域度_magnitude"] = ratio                 # 以 _magnitude 结尾，不计入残差
    out["零模是否定域（峰/均值 > 5）"] = 0.0 if ratio > 5.0 else 1.0
    return out


# ------------------------------------------------------------------------------
# 时间演化：把静态谱变成真的动力学
# ------------------------------------------------------------------------------
def _h2(k, m, r):
    """2×2 块 H(k) = σ_z sin k + σ_x (m + r(1−cos k))。"""
    a = float(np.sin(k))
    b = float(m + r * (1.0 - np.cos(k)))
    return np.array([[a, b], [b, -a]], dtype=complex)


def evolve_fft(N: int, m: float, r: float, psi0, dt: float, n_steps: int = 1):
    """周期边界、**常数质量**：k 空间精确演化（每步每 k 一个 2×2 指数）。

    ψ 的布局与 `build` 一致：``psi.reshape(N, 2)``，第 0/1 列是两个旋量分量。
    这是本模块第一处**真的时间演化**：``ψ(t+dt) = e^{−iHdt}ψ(t)``，指数精确、无分裂误差。
    """
    psi = np.asarray(psi0, dtype=complex).reshape(N, 2)
    k = 2.0 * np.pi * np.arange(N) / N
    for _ in range(n_steps):
        pk = np.fft.fft(psi, axis=0)
        out = np.empty_like(pk)
        for j in range(N):
            h = _h2(k[j], m, r)
            w, v = np.linalg.eigh(h)
            ph = np.exp(-1j * w * dt)
            out[j] = v @ (ph * (v.conj().T @ pk[j]))
        psi = np.fft.ifft(out, axis=0)
    return psi.reshape(2 * N)


def evolve_eigh(H, psi0, dt: float, n_steps: int = 1):
    """任意（含空间变化质量的）厄米 H 的精确演化，靠本征分解：``V e^{−iΛdt} V†``。"""
    w, v = np.linalg.eigh(H)
    psi = np.asarray(psi0, dtype=complex)
    ph = np.exp(-1j * w * dt)
    for _ in range(n_steps):
        psi = v @ (ph * (v.conj().T @ psi))
    return psi


def momentum_state(N: int, k: float, m: float, r: float, branch: int = +1):
    """H(k) 的本征态（branch=+1 取正能支），在实空间展开成 2N 向量。"""
    h = _h2(k, m, r)
    w, v = np.linalg.eigh(h)
    vec = v[:, int(np.argmax(w * branch))]
    x = np.arange(N)
    psi = np.zeros((N, 2), dtype=complex)
    psi[:, :] = np.exp(1j * k * x)[:, None] * vec[None, :]
    return (psi / np.linalg.norm(psi)).reshape(2 * N)


def packet(N: int, x0: float, k0: float, sigma: float, m: float, r: float, branch: int = +1):
    """高斯波包（中心 x0、平均动量 k0、宽度 sigma），旋量部分取 H(k0) 的正能本征矢。"""
    h = _h2(k0, m, r)
    w, v = np.linalg.eigh(h)
    vec = v[:, int(np.argmax(w * branch))]
    x = np.arange(N, dtype=float)
    g = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)
    psi = g[:, None] * vec[None, :]
    return (psi / np.linalg.norm(psi)).reshape(2 * N)


def position_expect(psi, N: int) -> float:
    """环上的位置期望（用 ⟨e^{i2πx/N}⟩ 的相位取，避开绕圈）。"""
    p = np.asarray(psi, dtype=complex).reshape(N, 2)
    dens = np.sum(np.abs(p) ** 2, axis=1)
    ang = np.sum(dens * np.exp(2j * np.pi * np.arange(N) / N))
    return float(np.angle(ang) * N / (2 * np.pi) % N)


def check_evolution(N: int = 64, m: float = 0.4, r: float = 0.0) -> dict:
    """P25：1+1D 时间演化——幺正性、色散相位、两套实现对拍、群速度。"""
    dt, n = 0.01, 50
    rng = np.random.default_rng(21)
    psi0 = rng.normal(size=2 * N) + 1j * rng.normal(size=2 * N)
    psi0 /= np.linalg.norm(psi0)

    a = evolve_fft(N, m, r, psi0, dt, n)
    H = build(N, m, r, periodic=True)
    b = evolve_eigh(H, psi0, dt, n)
    out = {
        "幺正性 |‖ψ(t)‖−1|": abs(float(np.linalg.norm(a)) - 1.0),
        "FFT vs 本征分解": float(np.abs(a - b).max()),
    }

    # 动量本征态：演化 t 后应等于 e^{−iE(k)t}·ψ(0)
    k0 = 2.0 * np.pi * 7 / N          # 必须是环上的离散动量，否则 e^{ikx} 不是本征态
    psi_k = momentum_state(N, k0, m, r)
    E = float(np.linalg.eigvalsh(_h2(k0, m, r))[1])
    t = dt * n
    out["色散相位 e^{−iE(k)t}"] = float(np.abs(evolve_eigh(H, psi_k, t) - np.exp(-1j * E * t) * psi_k).max())

    # 群速度：波包中心位移 vs v_g·t（v_g = dE/dk，用数值差分）
    sig, x0 = 6.0, N // 2
    p0 = packet(N, x0, k0, sig, m, r)
    t_g = 2.0
    dk = 1e-6
    vg = (float(np.sqrt(np.sin(k0 + dk) ** 2 + (m + r * (1 - np.cos(k0 + dk))) ** 2))
          - float(np.sqrt(np.sin(k0 - dk) ** 2 + (m + r * (1 - np.cos(k0 - dk))) ** 2))) / (2 * dk)
    dx = position_expect(evolve_eigh(H, p0, t_g), N) - position_expect(p0, N)
    dx = (dx + N / 2) % N - N / 2          # 解绕圈
    out["群速度_magnitude"] = abs(dx) / max(abs(vg * t_g), 1e-12)
    return out


# ------------------------------------------------------------------------------
# U(1) 规范场：纯规范不变性、Aharonov–Bohm 周期、Klein 隧穿
# ------------------------------------------------------------------------------
def transmission(m: float, V0: float, E: float) -> float:
    """1+1D 阶跃势的透射率（解析 transfer matrix）。

    H = −iσ_z∂_x + mσ_x + V(x)，两侧常数。给定能量 E，令 ε = E − V：
    动量为 k = √(ε² − m²)，旋量取 ``(ε − k, m)``（右行）与 ``(ε + k, m)``（左行）。
    两个方程解出反射幅度 B，T = 1 − |B|²。

    对 **m = 0** 自动给出 B = 0（因为入射/透射旋量的第一分量都是 0），
    这就是 1+1D 的 Klein 隧穿：**无质量粒子穿过任意阶跃势的透射率恰为 1**。
    """
    eps_L, eps_R = float(E), float(E - V0)
    if eps_L ** 2 < m * m or eps_R ** 2 < m * m:
        raise ValueError("两侧都必须有传播态（|ε| ≥ |m|）")
    k_L = float(np.sqrt(max(eps_L ** 2 - m * m, 0.0)))
    k_R = float(np.sqrt(max(eps_R ** 2 - m * m, 0.0)))
    # 旋量 (u, d) = (ε − k, m) 右行；(ε + k, m) 左行
    u_in, d_in = eps_L - k_L, m
    u_ref, d_ref = eps_L + k_L, m
    u_tr, d_tr = eps_R - k_R, m
    # 匹配：A(u_in) + B(u_ref) = C(u_tr)，对两个分量各一个方程，取 A = 1
    #   d: d_in + B d_ref = C d_tr
    #   u: u_in + B u_ref = C u_tr
    # 消掉 C：u_in d_tr + B u_ref d_tr = (d_in + B d_ref) u_tr
    denom = u_ref * d_tr - d_ref * u_tr
    numer = d_in * u_tr - u_in * d_tr
    if abs(denom) < 1e-300:
        return 1.0 if abs(numer) < 1e-300 else 0.0
    B = numer / denom
    return float(max(0.0, 1.0 - abs(B) ** 2))


def check_gauge(N: int = 64, m: float = 0.3, r: float = 1.0) -> dict:
    """P26：1+1D 的 U(1) 规范场——纯规范不变性、AB 周期、Klein 隧穿。"""
    out = {}

    # (a) 纯规范：开边界上 θ_x = φ_{x+1} − φ_x 必须给出与 θ=0 完全相同的谱
    rng = np.random.default_rng(31)
    phi = rng.normal(size=N)
    theta_pure = np.diff(phi)
    e0 = np.linalg.eigvalsh(build(N, m, r, periodic=False))
    e1 = np.linalg.eigvalsh(build(N, m, r, periodic=False, theta=theta_pure))
    out["纯规范变换不变性"] = float(np.abs(e0 - e1).max())

    # (b) AB 周期：环上均匀磁通，每键相位 +2π/N 等价于动量平移一格 ⇒ 谱不变
    th0 = np.zeros(N)
    th1 = np.full(N, 2.0 * np.pi / N)
    a0 = np.linalg.eigvalsh(build(N, m, r, periodic=True, theta=th0))
    a1 = np.linalg.eigvalsh(build(N, m, r, periodic=True, theta=th1))
    out["AB 周期 E(θ) = E(θ+2π/N)"] = float(np.abs(a0 - a1).max())

    # (c) Klein 隧穿：m=0 时透射率恰为 1；有质量时明显小于 1
    t0 = transmission(0.0, 0.5, 1.0)
    out["Klein: T(m=0) = 1"] = abs(t0 - 1.0)
    t1 = transmission(0.99, -0.3, 1.0)   # 质量接近 E：反射明显，T ≈ 0.66
    out["有质量时 T < 1_magnitude"] = t1
    return out


# ------------------------------------------------------------------------------
# 二次量子化的手征荷：为什么它给不出反常
# ------------------------------------------------------------------------------
def sea_chiral(N: int = 32, m: float = 0.3, r: float = 1.0, theta: float = 0.0) -> float:
    """把负能级全部填满（Dirac 海），求总手征荷 Q5 = Σ_{E<0} ⟨γ5⟩（γ5 = σ_z）。

    这就是“二次量子化”的最朴素写法：Q5 是占据态上单粒子手征荷之和。
    """
    H = build(N, m, r, periodic=True, theta=np.full(N, theta))
    w, v = np.linalg.eigh(H)
    g5 = np.kron(np.eye(N), SZ)
    occ = w < 0.0
    return float(sum(np.real(v[:, j].conj() @ g5 @ v[:, j]) for j in range(len(w)) if occ[j]))


def check_sea_chiral(N: int = 32, masses=(0.1, 0.3, 1.0), r: float = 1.0) -> dict:
    """P28：二次量子化的手征荷在磁通下**不流**——反常不是海的性质。

    对任意质量把磁通 θ 扫满一周 0→2π，海的手征荷 Q5(θ) 精确为零：
    粒子–空穴对称 + 手征对称让正负能级的 ⟨γ5⟩ 逐对抵消，扫磁通只是把谱整体平移，
    不改变任何占据态的手征荷。所以 **「填满的 Dirac 海」给不出反常**：
    反常是正规化的效应（正规化子不解耦 / 测度），三维的对应结论见 P18。

    这一项把该事实钉成可复跑的判据（残差 = 扫全周的最大 |Q5|）。
    """
    worst = 0.0
    for m in masses:
        for k in range(8):
            worst = max(worst, abs(sea_chiral(N, m, r, theta=2.0 * np.pi * k / 8)))
    return {"海手征荷 Q5(θ) ≡ 0（磁通全周）": worst}


# ------------------------------------------------------------------------------
# 反常的动力学面：磁通匀速扫过时的对产生（Landau–Zener）
# ------------------------------------------------------------------------------
def pair_production(Q: float = 1.5, m: float = 0.15, omega: float = 0.01,
                    n_steps: int = 30000) -> float:
    """磁通匀速扫过时，一个动量模式的**对产生概率**（Landau–Zener）。

    均匀磁通下动量守恒，每个 k 是一个独立的二能级问题：
        H(q) = σ_z sin q + σ_x m,    q = θ/N = ω·τ
    从 q = −Q 扫到 +Q（只跨过一个交叉点，|Q| < π/2 保证 sin 单调），初态取瞬时基态，
    测最终处于瞬时激发态的概率。

    解析预期：`P_LZ = exp(−π m² / ω)`（对角斜率 2ω、能隙 2m 的标准 LZ）。
    这就是 1+1D 反常的动力学面——真空在电场下不稳定，对产生率 ∝ exp(−πm²/eE)。
    """
    def eig(q):
        a, b = np.sin(q), m
        E = np.sqrt(a * a + b * b)
        return a, b, E

    T = Q / omega
    dt = 2.0 * T / n_steps
    a, b, _ = eig(-Q)
    # 初态：q=−Q 处的瞬时基态（负能支）
    _, v = np.linalg.eigh(np.array([[a, b], [b, -a]], dtype=complex))
    psi = v[:, 0].astype(complex)
    SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    for i in range(n_steps):
        tau = -T + (i + 0.5) * dt
        a, b, E = eig(omega * tau)
        if E < 1e-14:
            psi = psi
            continue
        c, s_ = np.cos(E * dt), np.sin(E * dt) / E
        U = c * np.eye(2, dtype=complex) - 1j * s_ * (a * SZ + b * SX)
        psi = U @ psi
    a, b, _ = eig(Q)
    _, v = np.linalg.eigh(np.array([[a, b], [b, -a]], dtype=complex))
    return float(abs(v[:, 1].conj() @ psi) ** 2)


def check_pair_production() -> dict:
    """P30：对产生（Landau–Zener）——反常的动力学面，带一个可测的时间尺度。

    P(ω) 随扫过速率 ω 按 `exp(−πm²/ω)` 增长：ω→0（绝热）P→0，ω 大（突然）P→1。
    判据取三个「离解析式可测」的点，比值偏离 < 15%；另报极小 P 处的数值地板。
    """
    out = {}
    for m, om in ((0.15, 0.01), (0.15, 0.03), (0.5, 0.08)):
        P = pair_production(1.5, m, om, 40000)
        Plz = float(np.exp(-np.pi * m * m / om))
        out[f"LZ 比值 (m={m}, ω={om})"] = max(0.0, abs(P / Plz - 1.0) - 0.15)
        out[f"P (m={m}, ω={om})_magnitude"] = P
        out[f"P_LZ (m={m}, ω={om})_magnitude"] = Plz
    out["数值地板 P (m=0.5, ω=0.01)_magnitude"] = pair_production(1.5, 0.5, 0.01, 30000)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check_dispersion(), ensure_ascii=False, indent=2))
    print(json.dumps(check_domain_wall(), ensure_ascii=False, indent=2))
    print(json.dumps(check_evolution(), ensure_ascii=False, indent=2))
    print(json.dumps(check_gauge(), ensure_ascii=False, indent=2))
    print(json.dumps(check_sea_chiral(), ensure_ascii=False, indent=2))
    print(json.dumps(check_pair_production(), ensure_ascii=False, indent=2))
