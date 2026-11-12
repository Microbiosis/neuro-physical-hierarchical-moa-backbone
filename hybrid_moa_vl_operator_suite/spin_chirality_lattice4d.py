# -*- coding: utf-8 -*-
"""spin_chirality.lattice4d —— 3+1D 格点指标：FLSG 瞬子背景 + overlap 算符 + FFT-Schur 惯性法。

物理依据
--------
指标定理（Atiyah–Singer，欧氏 4D）：index(D_ov) = 瞬子数 Q。
overlap 算符 D_ov = ½[(1+am) + (1−am) γ⁵ sign(γ⁵A)]，核 A = aD_W − (1+s)，
满足 Ginsparg–Wilson：{D,γ⁵} = 2 D γ⁵ D。

算法（两段，互为独立判据）
--------------------------
1. **固定背景读指标**（本文件 index_by_schur）：把 8N⁴ 维的 H = γ⁵A 分解成
   「自由核 + 只作用在核心站点上的修正」：
       H = H_c + Δ，‖Δ‖ ≤ η（站点局部 Weyl 界）
       n₋(H_c − σ) = n₋(H_0 − σ) + n₋(G_σ⁻¹ + E_C) − n₋(G_σ)
   其中 G_σ = P(H_0−σ)⁻¹P† 由动量空间逐 k 求逆 + 四维 FFT 得到，E_C 只在
   核心站点上非零（瞬子核心区，规模与 N 无关）。于是两次 σ = ±(η+ε) 给出
   index 的上下界；两端一致即精确值，不一致时用 Lanczos 把窗口内的本征值
   符号定下来。
2. **沿路径跟踪谱流**（spectral_min）：把链接按 U(t)=exp(t·log U) 连续打开，
   在若干 t 上量 H 的最小 |λ|；有本征值穿过零才可能改变指标。

依赖
----
numpy（必需）+ scipy（稀疏矩阵与 Lanczos）。缺 scipy 时本模块整体不可用，
`spin_chirality.lattice4d` 会给出明确提示。

数值规模（本机实测，见 INDEX_AUDIT.md）
----------------------------------------
N=6 dim=10368、N=10 dim=80000、N=12 dim=165888、N=20 dim=1.28M；
FFT-Schur 用时 8 s（N=6）→ 522 s（N=20），核心维 5240 起不再增长、内存约 0.4 GB。
"""

from __future__ import annotations

import itertools

import numpy as np

try:                                    # scipy 是可选依赖（extra: lattice）
    import scipy.sparse as sp
    from scipy.linalg import expm
    from scipy.sparse.linalg import eigsh

    HAS_SCIPY = True
except ImportError:                     # pragma: no cover
    sp = None
    expm = None
    eigsh = None
    HAS_SCIPY = False

S1 = np.array([[0, 1], [1, 0]], dtype=complex)
S2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
S3 = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# FLSG 瞬子构造用的两组 Pauli 组合
SS = (1j * S1, 1j * S2, 1j * S3, I2)
SD = (-1j * S1, -1j * S2, -1j * S3, I2)

# Weyl 基（与 operator.py 的 γ⁵ = diag(−1,−1,+1,+1) 同一约定）
_Z2 = np.zeros((2, 2), complex)
G1, G2, G3, G4 = (np.block([[_Z2, U], [U.conj().T, _Z2]]) for U in (1j * S1, 1j * S2, 1j * S3, I2))
GAMMA = (G1, G2, G3, G4)
G5 = np.diag(np.array([-1.0, -1.0, 1.0, 1.0], complex))

I8 = np.eye(8, dtype=complex)
I_LINK = np.eye(2, dtype=complex)



__all__ = [
    "flsg_links",
    "link_unit_residual",
    "build_kernel",
    "gamma5_matrix",
    "gamma5_herm_residual",
    "free_H_k",
    "free_inertia",
    "free_index",
    "free_green",
    "eta_bound",
    "core_sites",
    "core_block",
    "inertia",
    "index_by_schur",
    "spectral_min",
    "check_index_4d",
]

def _require_scipy():
    if not HAS_SCIPY:
        raise ImportError("spin_chirality.lattice4d 需要 scipy（pip install 'spin-chirality[lattice]'）")


# =============================================================== 瞬子背景
def flsg_links(N: int, R: float, sign: int = +1) -> dict:
    """周期环面上的 FLSG 单瞬子链接（Fox–Laursen–Schierholz–Gilchrist 构造）。

    U_μ(x) = exp[ −τ_μν (x−c)_ν · R² / (2 n² (n²+R²)) ]，n = x − c，c 为格心。
    sign = +1 是瞬子（指标 −1），−1 是反瞬子（指标 +1）。
    """
    _require_scipy()
    center = [N / 2.0 + 0.5] * 4
    links = {}
    for s in itertools.product(range(N), repeat=4):
        for mu in range(4):
            nv = np.array([s[i] - center[i] for i in range(4)], float)
            n2 = float(nv @ nv)
            tau = np.zeros((2, 2), complex)
            for nu in range(4):
                if nu == mu:
                    continue
                smunu = (SD[mu] @ SS[nu] - SD[nu] @ SS[mu]) if sign == +1 \
                    else (SS[mu] @ SD[nu] - SS[nu] @ SD[mu])
                tau += smunu * nv[nu]
            links[(s, mu)] = expm(-tau * R ** 2 / (2.0 * n2 * (n2 + R ** 2)))
    return links


def link_unit_residual(links) -> float:
    """所有链接与单位的最大偏差（平凡背景应为 0）。"""
    return max(float(np.max(np.abs(U - I_LINK))) for U in links.values())


# =============================================================== 稀疏算符
def build_kernel(N: int, links=None, m: float = 0.0, r: float = 1.0):
    """稀疏 Wilson–Dirac 核算符 A = D_W − 1（8N⁴ 维，SU(2) 色 × Dirac 4）。

    正向 D_W(s, s+μ) = γ_μ U_μ(s)/2 − r/2；反向用**同一条链接**的共轭，
    因此 γ⁵Aγ⁵ = A† 精确成立（H = γ⁵A 厄米）。
    """
    _require_scipy()
    sites = list(itertools.product(range(N), repeat=4))
    ix = {s: i for i, s in enumerate(sites)}
    V = len(sites)
    U = np.empty((V, 4, 2, 2), complex)
    nb = np.empty((V, 4), np.int64)
    for si, s in enumerate(sites):
        for mu in range(4):
            U[si, mu] = links[(s, mu)] if links is not None else I_LINK
            t = list(s)
            t[mu] = (t[mu] + 1) % N
            nb[si, mu] = ix[tuple(t)]

    Gam = np.array(GAMMA)
    blk_f = np.einsum("mij,smab->smiajb", Gam, U).reshape(V, 4, 8, 8) / 2 - (r / 2) * I8
    blk_b = -np.einsum("mij,smab->smiajb", Gam, U.conj().transpose(0, 1, 3, 2)).reshape(V, 4, 8, 8) / 2 - (r / 2) * I8

    a_idx = np.arange(8)
    Varr = np.arange(V)
    rows = [np.broadcast_to(8 * Varr[:, None] + a_idx[None, :], (V, 8)).ravel()]
    cols = [rows[0]]
    vals = [np.tile(np.diag((m - 1.0 + 4 * r) * I8), V)]

    for mu in range(4):
        rows.append(np.broadcast_to(8 * Varr[:, None, None] + a_idx[None, :, None], (V, 8, 8)).ravel())
        cols.append(np.broadcast_to(8 * nb[:, mu][:, None, None] + a_idx[None, None, :], (V, 8, 8)).ravel())
        vals.append(blk_f[:, mu].ravel())
        rows.append(np.broadcast_to(8 * nb[:, mu][:, None, None] + a_idx[None, :, None], (V, 8, 8)).ravel())
        cols.append(np.broadcast_to(8 * Varr[:, None, None] + a_idx[None, None, :], (V, 8, 8)).ravel())
        vals.append(blk_b[:, mu].ravel())

    dim = 8 * V
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(dim, dim), dtype=complex).tocsr()


def gamma5_matrix(N: int):
    """全局 γ⁵ = I_{N⁴} ⊗ (γ⁵ ⊗ I₂)（本表示里 γ⁵ 对角）。"""
    _require_scipy()
    local = sp.csr_matrix(np.kron(G5, I2))
    return sp.kron(sp.eye(N ** 4, format="csr", dtype=complex), local, format="csr")


def gamma5_herm_residual(N: int, links=None) -> float:
    """‖γ⁵Aγ⁵ − A†‖∞（H = γ⁵A 厄米的前提，正确构造应为 0）。"""
    _require_scipy()
    A = build_kernel(N, links)
    g = gamma5_matrix(N)
    return float(abs(g @ A @ g - A.conj().T).max())


# =============================================================== 自由核
def free_H_k(k, m: float = 0.0, r: float = 1.0) -> np.ndarray:
    """自由核在动量 k 处的 8×8 块：H_0(k) = γ⁵ A(k) ⊗ I₂。"""
    A = np.zeros((4, 4), complex)
    for mu in range(4):
        A += GAMMA[mu] * 1j * np.sin(k[mu])
    A += (m - 1.0 + r * sum(1 - np.cos(k[mu]) for mu in range(4))) * np.eye(4)
    return np.kron(G5 @ A, I2)


def free_inertia(N: int, sigma: float = 0.0, m: float = 0.0, r: float = 1.0) -> int:
    """n₋(H_0 − σ)：逐动量数 8×8 块的负本征值个数。"""
    ks = 2 * np.pi * np.arange(N) / N
    n_neg = 0
    for idx in itertools.product(range(N), repeat=4):
        k = np.array([ks[i] for i in idx])
        w = np.linalg.eigvalsh(free_H_k(k, m, r) - sigma * np.eye(8))
        n_neg += int(np.sum(w < 0))
    return n_neg


def free_index(N: int) -> float:
    """自由核的指标（谱对称 ⇒ 0），用作链路的自洽性前提。"""
    return free_inertia(N, 0.0) - 4 * N ** 4


def free_green(N: int, sigma: float, m: float = 0.0, r: float = 1.0) -> np.ndarray:
    """G_σ(d) = (1/V)Σ_k e^{ik·d}[M(k) − σI]⁻¹，返回 (N,N,N,N,8,8)。"""
    ks = 2 * np.pi * np.arange(N) / N
    Mk = np.zeros((N, N, N, N, 8, 8), complex)
    for idx in itertools.product(range(N), repeat=4):
        k = np.array([ks[i] for i in idx])
        Mk[idx] = np.linalg.inv(free_H_k(k, m, r) - sigma * I8)
    G = np.fft.ifftn(Mk, axes=(0, 1, 2, 3))
    return G


# =============================================================== 核心与界
def eta_bound(links, strong, N: int) -> float:
    """‖H − H_c‖₂ 的上界（站点局部 Weyl 界）：η = ½ max_x[Σ(a+b) + √Σ(a−b)²]。"""
    a = {(s, mu): (0.0 if strong[(s, mu)] else float(np.linalg.norm(links[(s, mu)] - I_LINK, 2)))
         for s in itertools.product(range(N), repeat=4) for mu in range(4)}

    def sh(s, mu, d):
        t = list(s)
        t[mu] = (t[mu] + d) % N
        return tuple(t)

    eta = 0.0
    for s in itertools.product(range(N), repeat=4):
        A_ = [a[(s, mu)] for mu in range(4)]
        B_ = [a[(sh(s, mu, -1), mu)] for mu in range(4)]
        eta = max(eta, 0.5 * (sum(A_) + sum(B_) + np.sqrt(sum((A_[k] - B_[k]) ** 2 for k in range(4)))))
    return float(eta)


def core_sites(strong, N: int):
    """强链接的两个端点（只取起点会漏掉扰动支撑）。"""
    def sh(s, mu, d):
        t = list(s)
        t[mu] = (t[mu] + d) % N
        return tuple(t)

    core = set()
    for (s, mu), st in strong.items():
        if st:
            core.add(s)
            core.add(sh(s, mu, +1))
    return sorted(core)


def core_block(G: np.ndarray, core, N: int) -> np.ndarray:
    """把 G(d) 组装成核心×核心的稠密块 (8|core|, 8|core|)。"""
    nc = len(core)
    out = np.zeros((8 * nc, 8 * nc), complex)
    for p, sp_ in enumerate(core):
        for q, sq in enumerate(core):
            d = tuple((sp_[i] - sq[i]) % N for i in range(4))
            out[8 * p:8 * p + 8, 8 * q:8 * q + 8] = G[d]
    return out


def inertia(M: np.ndarray) -> int:
    """稠密 Hermite 矩阵的负本征值个数。"""
    w = np.linalg.eigvalsh((M + M.conj().T) / 2)
    return int(np.sum(w < 0))


# =============================================================== 指标主流程
def index_by_schur(N: int, R: float = 2.5, thr: float = 0.18, eps: float = 1e-6,
                   sign: int = +1, verbose: bool = False) -> dict:
    """FFT-Schur 惯性法：给出 FLSG 瞬子背景下的 4D 指标与其区间。

    返回 dict：index、eta、核心维、两端是否一致、是否用 Lanczos 定符号。
    """
    _require_scipy()
    links = flsg_links(N, R, sign)
    strong = {k: (np.max(np.abs(U - I_LINK)) > thr) for k, U in links.items()}
    links_c = {k: (U if strong[k] else I_LINK) for k, U in links.items()}
    core = core_sites(strong, N)
    eta = eta_bound(links, strong, N)

    G5s = gamma5_matrix(N)
    H_0 = (G5s @ build_kernel(N, {k: I_LINK for k in links})).tocsr()
    H_c = (G5s @ build_kernel(N, links_c)).tocsr()
    dim = H_0.shape[0]
    ix = {s: i for i, s in enumerate(itertools.product(range(N), repeat=4))}
    idx = np.concatenate([8 * ix[s] + np.arange(8) for s in core])
    E_C = (H_c - H_0)[idx][:, idx].toarray()
    del H_0, H_c

    res = {}
    for sgn in (+1, -1):
        sig = sgn * (eta + eps)
        Gc = core_block(free_green(N, sig), core, N)
        n_G = inertia(Gc)
        M = np.linalg.inv(Gc) + E_C
        n_core = inertia(M)
        del Gc, M
        res[sgn] = free_inertia(N, sig) + n_core - n_G - dim / 2.0

    resolved = None
    if abs(res[+1] - res[-1]) >= 0.5:            # 区间不闭合：用 Lanczos 定窗口内符号
        k_win = int(abs(res[+1] - res[-1])) + 2
        Hsp = (G5s @ build_kernel(N, links)).tocsr()
        vals, vecs = eigsh((Hsp @ Hsp).tocsr(), k=k_win, which="SA")
        Msub = vecs.conj().T @ (Hsp @ vecs)
        w = np.sort(np.linalg.eigvalsh((Msub + Msub.conj().T) / 2))
        lo = max(abs(eta + eps), 0.0)
        in_win = w[(w > -lo) & (w < lo)]
        n_neg_H = int(round(res[-1] + dim / 2.0)) + int(np.sum(in_win < 0))
        resolved = n_neg_H - dim / 2.0

    ok = abs(res[+1] - res[-1]) < 0.5
    index = res[+1] if ok else (resolved if resolved is not None else res[+1])
    if verbose:
        print(f"  N={N} R={R} sign={sign:+d}: dim={dim} core={len(core)}站点/{len(idx)}维 "
              f"强链接={sum(strong.values())} η={eta:.4f}")
        print(f"    σ=+{eta + eps:.4f}: index={res[+1]:+.1f}   σ=−{eta + eps:.4f}: index={res[-1]:+.1f}"
              f"  ⇒ {index:+.1f}（{'区间闭合' if ok else 'Lanczos 定符号'}）")
    return {"index": float(index), "eta": eta, "core_dim": int(len(idx)),
            "bracket_ok": bool(ok), "resolved": resolved is not None}


def spectral_min(N: int, links, ts=(0.2, 0.4, 0.6, 0.8, 1.0), k: int = 3):
    """沿 U(t)=exp(t·log U) 把背景连续打开，给出各 t 的最小 k 个 |λ(H)|。

    有本征值穿过零才可能改变指标：|λ| 离开 0 的谷底越小、越接近 0，越可能发生穿越。
    """
    _require_scipy()
    g = gamma5_matrix(N)
    out = []
    for t in ts:
        scaled = {}
        for key, U in links.items():
            w, v = np.linalg.eig(U)
            scaled[key] = v @ np.diag(np.exp(t * np.log(w + 0j))) @ np.linalg.inv(v)
        H = (g @ build_kernel(N, scaled)).tocsr()
        vals = eigsh((H @ H).tocsr(), k=k, which="SA", return_eigenvectors=False)
        out.append((float(t), np.sort(np.sqrt(np.clip(vals, 0, None)))))
    return out


def check_index_4d(N: int = 8, R: float = 2.5, thr: float = 0.18) -> dict:
    """3+1D 指标检验：FLSG 单瞬子应给 index = −1，反瞬子应给 +1。

    返回三项残差（全为 0 即通过）：瞬子指标、反瞬子指标、以及 γ⁵-Hermiticity 残差。
    """
    inst = index_by_schur(N, R, thr=thr, sign=+1)
    anti = index_by_schur(N, R, thr=thr, sign=-1)
    return {
        "index_instanton_vs_-1": abs(inst["index"] + 1.0),
        "index_anti_vs_+1": abs(anti["index"] - 1.0),
        "gamma5_herm": gamma5_herm_residual(N, flsg_links(N, R, +1)),
    }


if __name__ == "__main__":              # pragma: no cover
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    print(f"dim = {8 * n ** 4}")
    result = index_by_schur(n, r, verbose=True)
    print(result)
    # FLSG 单瞬子背景的指标应为 −1；反瞬子为 +1。
    # bracket_ok 只表示「参数区间是否闭合」——未闭合时改用 Lanczos 定符号，
    # index 本身仍是正确的，故判定只看 index。
    _index = float(result["index"])
    _ok = abs(_index + 1.0) < 1e-9
    print(f"[{'PASS' if _ok else 'FAIL'}] 3+1D Wilson-Dirac index = {_index:+.1f} "
          f"(expect -1; bracket_closed={result.get('bracket_ok')})")
    sys.exit(0 if _ok else 1)
