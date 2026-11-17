# -*- coding: utf-8 -*-
"""1+1D Kogut–Susskind 质量隙的连续极限：MPS/DMRG（N 到 60、x 到 0.15）。

为什么要换方法
--------------
`ks_ed.py` 的稠密 ED 到 N ≤ 14；`ks_lanczos.py` 的矩阵自由 Lanczos 到 N = 24，
三重外推给 G(0) = gap/e = 0.594 ± 0.073（比 e/√π = 0.5641895 高 5.2%）。
要把 5% 的差异判死，需要 x = ea ≤ 0.156 且 m_γL ≥ 5，即 N ≈ 57；而稠密基矢
4^57 ≈ 1e34。MPS 的代价只随 N 线性、随键维三次，与基矢维数无关——这就是本模块。

做法
----
① 高斯定律 `E_x = Σ_{y≤x}ρ_y`（θ=0 扇区，环上切口处 q=0）把电场整个消掉，H 变成一条
   局域自旋链（d = 2，交错费米子 ↔ 自旋，Jordan–Wigner）：

       H = (g²/2)Σ_x E_x² + mΣ_x(−1)^x n_x − t Σ_x (S⁺_xS⁻_{x+1} + h.c.) − t·(跨切口项)

   跨切口项带一条 JW 弦 `Π_{1≤y≤N−2}(−σ^z_y)`——环与开链的唯一区别。
② 库仑项展开成对和 `(g²/2)Σ_{y,z} ρ_yρ_z (N−max(y,z))`（用 Σρ = 0 化简），于是 MPO
   只要 8 个通道，**且不对电场谱做任何截断**（ED 的 `|E| ≤ Emax` 只是让基矢有限的
   数值手段，这里不需要）。
③ 两格点 DMRG：局部用 Lanczos 求最低态；激发态用"与基态正交的投影 Lanczos"，
   正交方向由 ψ0/ψ1 的重叠环境给出（Jeckelmann/正交 DMRG 的做法）。
④ `⟨ρ_rρ_0⟩` 用一次左扫 O(Nχ³) 算完，累加成 `⟨E(x)E(0)⟩_conn`，给第二个 m_γ。

约定与标定
----------
t = 0.5、a = 1 ⇒ e² = 2g²t ⇒ **e = g、x = ea = g**，所以 `gap/e = gap/g`。
两把独立的尺钉住归一化（与 P35 相同）：强耦合 gap → g²/2、自由极限
E0 = Σ 最低 N/2 个 ±2t|cos(k/2)|（PBC）。

与 `ks_ed.py` 的对齐（P36 判据，见 `check_ks_mps`）
---------------------------------------------------
N = 8/10/12/14、ED 侧取 Emax = 4/5（已饱和 = 无截断），E0 与 gap 逐个数字吻合到
≤ 4e-11（实测，见 OPERATOR_STATUS.md §6quaterdecies）。N = 16/24 上与
`ks_lanczos.py`（Emax=2，独立实现）的差 ≤ 4e-4 相对——那正是 Emax 截断本身的量级。

磁通（扭曲边界）
----------------
`mpo_ks(..., phi=φ)` 只改跨切口那一支：χ†_{N−1}χ_0 得 e^{+iφ}、其共轭得 e^{−iφ}。
φ = 0 时 dtype 仍是 float；φ ≠ 0 时 H 复厄米，DMRG / Lanczos / 罚项全部走复数分支
（`energy` 取实部、`lanczos` 用 v†w 重正交、`_add_penalty` 用 np.vdot）。
分辨"粒子 vs 弦态"就用它，见 `ks_photon.py`（P37）。

尺寸与代价（本机实测，χ=96、pen=1）
-----------------------------------
    N   :  24     32     40     48     60
    E0  :  60 s   76 s   95 s   115 s  140 s   （基态+激发态一次算完）
键维在 N = 60、x = 0.15 时只涨到 96（有隙 ⇒ 纠缠面积律），χ = 512 完全用不上。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "mpo_ks",
    "mps_random",
    "dmrg_ks",
    "dmrg_gap",
    "corr_rho",
    "corr_rho_from",
    "corr_electric",
    "one_site_all",
    "MPS_GAP_SCAN",
    "MPS_CHI_SCAN",
    "fit_continuum_gap_mps",
    "xi_from_corr",
    "check_ks_mps",
]

SQRT_PI_INV = float(1.0 / np.sqrt(np.pi))          # 0.5641895…，Schwinger 光子质量 e/√π

# 局域算符（bit = 1 表示占据）
_I2 = np.eye(2)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]])           # |1><0|，抬占据
_SM = _SP.T
_ZN = np.diag([1.0, -1.0])                         # (-1)^n = −σ^z


def _rho_op(x: int) -> np.ndarray:
    """ρ_x = (σ^z + (−1)^x)/2 的对角表示。"""
    s = (-1.0) ** x
    return np.diag([s * 0.5 - 0.5, s * 0.5 + 0.5])


# ==============================================================================
# 1. MPO
# ==============================================================================
def mpo_ks(N: int, g: float, m: float = 0.0, t: float = 0.5, pen: float = 0.0,
           phi: float = 0.0):
    """KS 哈密顿量的 MPO，返回 (list[W], D)，W[x] 形状 (D,2,2,D) = W[a,s,s',b]。

    通道：0=I、1=库仑对（已插一个 ρ）、2=S⁺、3=S⁻、4=跨切口（S⁻ 已插）、
    5=跨切口（S⁺ 已插）、6=Q² 罚项（Σρ 的平方，用来钉住电荷扇区）、7=D（收尾）。

    `phi` = 穿过环的磁通（扭曲边界条件）：跨切口那一支 χ†_{N−1}χ_0 得 e^{+iφ}、
    其厄米共轭得 e^{−iφ}。φ = 0 时 dtype 仍是 float（回归零代价），φ ≠ 0 时整个
    MPO 变复数（H 仍厄米：`mpo_dense` 的自检项 ‖H−H†‖ = 0）。

    库仑项按对和写：`Σ_{x=0}^{N−1} E_x² = Σ_{y,z} ρ_yρ_z (N−max(y,z))`，所以
    「插一个 ρ、走一段恒等、在 z 处以 2(N−z)ρ_z 收尾」就是全部。
    """
    D = 8
    out = []
    dt = complex if phi else float
    zph = np.exp(1j * phi) if phi else 1.0
    zmh = np.exp(-1j * phi) if phi else 1.0
    for x in range(N):
        W = np.zeros((D, 2, 2, D), dtype=dt)
        rho = _rho_op(x)
        W[0, :, :, 0] += _I2
        W[7, :, :, 7] += _I2
        W[0, :, :, 7] += m * ((-1.0) ** x) * np.diag([0.0, 1.0])
        W[0, :, :, 1] += rho
        W[1, :, :, 1] += _I2
        W[1, :, :, 7] += 0.5 * g * g * 2.0 * (N - x) * rho
        W[0, :, :, 7] += 0.5 * g * g * (N - x) * (rho @ rho)
        W[0, :, :, 6] += pen * rho
        W[6, :, :, 6] += _I2
        W[6, :, :, 7] += 2.0 * pen * rho
        W[0, :, :, 7] += pen * (rho @ rho)
        if x <= N - 2:
            W[0, :, :, 2] += -t * _SP
            W[0, :, :, 3] += -t * _SM
        if x >= 1:
            W[2, :, :, 7] += _SM
            W[3, :, :, 7] += _SP
        if x == 0:
            W[0, :, :, 4] += -t * zph * _SM
            W[0, :, :, 5] += -t * zmh * _SP
        if 0 < x < N - 1:
            W[4, :, :, 4] += _ZN
            W[5, :, :, 5] += _ZN
        if x == N - 1:
            W[4, :, :, 7] += _SP
            W[5, :, :, 7] += _SM
        out.append(W)
    return out, D


def mpo_dense(mpo, D):
    """把 MPO 展成稠密矩阵（只在小 N 上做，用于与 `ks_ed` 对拍）。

    dtype 跟随 W（φ ≠ 0 时是复数）：早期版本写死 float，会静默把 i 项扔掉。
    """
    dt = complex if np.iscomplexobj(mpo[0]) else float
    M = np.zeros((1, 1, D), dtype=dt)
    M[0, 0, 0] = 1.0
    for W in mpo:
        d0 = M.shape[0]
        M2 = np.zeros((2 * d0, 2 * d0, D), dtype=dt)
        for r in range(2):
            for rp in range(2):
                M2[r * d0:(r + 1) * d0, rp * d0:(rp + 1) * d0, :] = np.einsum(
                    'ija,ab->ijb', M, W[:, r, rp, :])
        M = M2
    return M[:, :, -1]


# ==============================================================================
# 2. 环境、有效哈密顿量、Lanczos
# ==============================================================================
def _update_left(L, W, A):
    LW = np.tensordot(L, W, axes=([0], [0]))            # (i,j,s,S,c)
    X = np.tensordot(LW, A, axes=([0, 2], [0, 1]))      # (j,S,c,k)
    return np.tensordot(X, A.conj(), axes=([0, 1], [0, 1]))


def _update_right(W, A, Rn):
    X = np.tensordot(A, Rn, axes=([2], [1]))            # (j,s,b,K)
    Y = np.tensordot(X, A.conj(), axes=([3], [2]))      # (j,s,b,J,S)
    return np.tensordot(W, Y, axes=([1, 2, 3], [1, 4, 2]))


def left_envs(mpo, A, upto=None):
    N = len(A)
    upto = N if upto is None else upto
    L = [None] * (N + 1)
    L[0] = np.zeros((mpo[0].shape[0], 1, 1))
    L[0][0, 0, 0] = 1.0
    for i in range(upto):
        L[i + 1] = _update_left(L[i], mpo[i], A[i])
    return L


def right_envs(mpo, A, from_=0):
    N = len(A)
    R = [None] * (N + 1)
    R[N] = np.zeros((mpo[0].shape[0], 1, 1))
    R[N][-1, 0, 0] = 1.0
    for i in range(N - 1, from_ - 1, -1):
        R[i] = _update_right(mpo[i], A[i], R[i + 1])
    return R


def apply_heff(L, W1, W2, R, Th):
    """两格点有效哈密顿量：

    Y[j,S,T,K] = Σ L[a,i,j] W1[a,s,S,c] W2[c,t,T,b] R[b,k,K] Θ[i,s,t,k]。
    四步全走 BLAS matmul（比长链 einsum 快 3–5 倍）。
    """
    D = W1.shape[0]
    cl, cr = Th.shape[0], Th.shape[3]
    P1 = (L.transpose(1, 2, 0).reshape(cl * cl, D) @ W1.reshape(D, 4 * D)
          ).reshape(cl, cl, 2, 2, D).transpose(1, 3, 0, 2, 4)
    P2 = (P1.transpose(0, 1, 4, 2, 3).reshape(cl * 2 * D, cl * 2) @ Th.reshape(cl * 2, 2 * cr)
          ).reshape(cl, 2, D, 2, cr)
    P3 = np.tensordot(P2.reshape(cl, 2, D * 2, cr), W2.reshape(D * 2, 2 * D),
                      axes=([2], [0])).reshape(cl, 2, cr, 2, D)
    P4 = P3.transpose(0, 1, 3, 2, 4).reshape(cl * 4, cr * D)
    return (P4 @ R.transpose(1, 0, 2).reshape(cr * D, cr)).reshape(cl, 2, 2, cr)


def energy(mpo, A):
    """⟨ψ|H|ψ⟩（A 需归一）。φ ≠ 0 时态是复的，取实部。"""
    return float(np.real(left_envs(mpo, A)[-1][-1, 0, 0]))


def lanczos(Aop, x0, iters=10, tol=1e-11):
    """最低本征对：两遍全重正交 + Ritz 值收敛提前退出。

    复数安全：α 取实部（厄米算符下虚部只是舍入），重正交用 v†w，Krylov 向量的
    dtype 由算符探一次定出（初态可能是实、算符是复）。
    """
    shape, n = x0.shape, x0.size
    nsteps = min(iters, n)
    probe = Aop(np.asarray(x0).reshape(shape)).ravel()
    V = np.zeros((nsteps, n), dtype=np.result_type(np.asarray(x0).dtype, probe.dtype))
    al, be = np.zeros(nsteps), np.zeros(nsteps)
    V[0] = x0.ravel() / np.linalg.norm(x0)
    prev, vec, used = np.inf, V[0], 1
    for j in range(nsteps):
        w = Aop(V[j].reshape(shape)).ravel()
        al[j] = np.real(np.vdot(V[j], w))
        if j > 0:
            w = w - be[j - 1] * V[j - 1]
        w = w - al[j] * V[j]
        for _ in range(2):
            w = w - V[:j + 1].T @ (V[:j + 1].conj() @ w)
        used = j + 1
        T = np.diag(al[:used])
        if used > 1:
            T += np.diag(be[:used - 1], 1) + np.diag(be[:used - 1], -1)
        ev, evec = np.linalg.eigh(T)
        vec = V[:used].T @ evec[:, 0]
        if abs(ev[0] - prev) < tol * max(1.0, abs(ev[0])):
            break
        prev = ev[0]
        b = np.linalg.norm(w)
        if b < 1e-14 or j == nsteps - 1:
            break
        be[j] = b
        V[j + 1] = w / b
    return prev, vec, used


# ==============================================================================
# 3. MPS 工具
# ==============================================================================
def mps_random(N: int, chi: int, seed: int = 0):
    """随机初始 MPS（键维按 2 的幂爬升再回落），左规范。"""
    rng = np.random.default_rng(seed)
    dims = [max(1, min(chi, 2 ** min(i, N - i))) for i in range(N + 1)]
    A = [rng.normal(size=(dims[i], 2, dims[i + 1])) for i in range(N)]
    for i in range(N - 1):
        l, d, r = A[i].shape
        q, rr = np.linalg.qr(A[i].reshape(l * d, r))
        A[i] = q.reshape(l, d, q.shape[1])
        A[i + 1] = np.tensordot(rr, A[i + 1], axes=([1], [0]))
    return A


def right_canonical(A):
    A = [a.copy() for a in A]
    for i in range(len(A) - 1, 0, -1):
        l, d, r = A[i].shape
        q, rr = np.linalg.qr(A[i].reshape(l, d * r).T)
        A[i] = q.T.reshape(q.shape[1], d, r)
        A[i - 1] = np.tensordot(A[i - 1], rr.T, axes=([2], [0]))
    return A


def _split(vec, l, cr, chi, right_side):
    """SVD 拆分两格点张量；右扫时把 S 留在左边，保证 A[i+1] 右等距。"""
    U, S, Vt = np.linalg.svd(vec.reshape(l * 2, 2 * cr), full_matrices=False)
    keep = max(1, min(chi, int((S > 1e-15 * S[0]).sum()) if S.size else 1))
    if right_side:
        return (U[:, :keep] * S[:keep]).reshape(l, 2, keep), Vt[:keep].reshape(keep, 2, cr)
    return U[:, :keep].reshape(l, 2, keep), (S[:keep, None] * Vt[:keep]).reshape(keep, 2, cr)


def _norm_right(A0):
    """ψ0 的右范数环境。"""
    N = len(A0)
    R0 = [None] * (N + 1)
    R0[N] = np.ones((1, 1))
    for i in range(N - 1, -1, -1):
        X = np.tensordot(A0[i], R0[i + 1], axes=([2], [0]))
        R0[i] = np.tensordot(X, A0[i].conj(), axes=([1, 2], [1, 2]))
    return R0


def _overlap_left(A0, LO, A1):
    """LO_new[k0,k1] = Σ A0[j0,s,k0] conj(A1[j1,s,k1]) LO[j0,j1]。"""
    X = np.tensordot(LO, A0, axes=([0], [0]))                 # (j1,s,k0)
    return np.tensordot(X, A1.conj(), axes=([0, 1], [0, 1]))  # (k0,k1)


def _overlap_right(A0, RO, A1):
    """RO_new[j0,j1] = Σ A0[j0,s,k0] conj(A1[j1,s,k1]) RO[k0,k1]。"""
    X = np.tensordot(A0, RO, axes=([2], [0]))                 # (j0,s,k1)
    return np.tensordot(X, A1.conj(), axes=([1, 2], [1, 2]))  # (j0,j1)


def _overlap_vec(A0, LO, RO, i):
    """ψ0 在 ψ1 当前规范下的两格点振幅 Φ0[a1L,s,t,a1R]（精确，不掺范数环境）。

    LO[i] 的指标次序是 (a0,a1)（`_overlap_left` 的定义），所以直接与 A0[i] 的
    左脚缩并就得到 a1L；右环境 RO[i+2] 的指标是 (a0,a1)，与 A0[i+1] 的右脚缩并
    得到 a1R。ψ1 右规范 ⇒ 右基正交，不需要 ψ0 的范数环境 R0。
    （早期版本在这里多插了一个 `_norm_right` 环境，实数下恰好仍指得准 ψ0，
    复数（φ ≠ 0）下就错开了——见 OPERATOR_STATUS.md §6quindecies。）
    """
    X = np.tensordot(LO, A0[i], axes=([0], [0]))              # (a1L,s,a0R)
    Y = np.tensordot(X, A0[i + 1], axes=([2], [0]))           # (a1L,s,t,a0RR)
    return np.tensordot(Y, RO[i + 2], axes=([3], [0]))        # (a1L,s,t,a1R)


def _add_penalty(op, phi, lam):
    """φ = ψ0 的两格点振幅；lam 为 None → 硬投影（精确正交），否则 λ|φ⟩⟨φ|/‖φ‖² 罚项。

    内积一律走 `np.vdot`（共轭在第一变元），所以复数态（磁通 φ ≠ 0）同样正确；
    只装 numpy 的实数态下与旧写法逐位相同。
    """
    nrm = float(np.real(np.vdot(phi, phi)))
    if lam is None:
        def f(v):
            v = v - phi * (np.vdot(phi, v) / nrm)
            y = op(v)
            return y - phi * (np.vdot(phi, y) / nrm)
        return f
    return lambda v: op(v) + (lam / nrm) * phi * np.vdot(phi, v)


# ==============================================================================
# 4. 两格点 DMRG
# ==============================================================================
def _normalize_excited(excited):
    """把 excited 归一为 MPS 列表：None→[]，单 MPS→[MPS]，列表→原样。"""
    if excited is None:
        return []
    if isinstance(excited[0], np.ndarray):
        return [excited]
    return list(excited)


def dmrg_ks(N: int, g: float, m: float = 0.0, chi: int = 64, sweeps: int = 12,
            t: float = 0.5, pen: float = 1.0, seed: int = 1, A_init=None,
            excited=None, lam=None, tol: float = 1e-14, iters: int = 12,
            verbose: bool = False, phi: float = 0.0):
    """两格点 DMRG。`excited` = 基态 MPS（或 MPS 列表）时求"与它们正交"的最低态。

    返回 {A, E, hist, chi}。电荷扇区用 `pen`·(Σρ)² 钉住（pen = 1 比能隙大两个量级，
    实测 ⟨Q²⟩ < 1e-20 的影响，见 check_ks_mps 的 "电荷扇区" 项）。

    `excited` 接受三种形态：None / 单个 MPS / MPS 列表。多态时按 Gram–Schmidt 逐个
    投影（lam = None 硬投影；否则加 λ 罚项），向后兼容既有的单态调用。
    """
    ex_list = _normalize_excited(excited)
    mpo, D = mpo_ks(N, g, m, t=t, pen=pen, phi=phi)
    A = mps_random(N, chi, seed=seed) if A_init is None else [a.copy() for a in A_init]
    A = right_canonical(A)
    Ns = len(A)
    hist = []
    Rset = right_envs(mpo, A)
    nex = len(ex_list)
    for sw in range(sweeps):
        L = left_envs(mpo, A)
        LOs = [[None] * (Ns + 1) for _ in range(nex)]
        ROs = [[None] * (Ns + 1) for _ in range(nex)]
        for k in range(nex):
            LOs[k][0] = np.ones((1, 1))
            for i in range(Ns - 1):
                LOs[k][i + 1] = _overlap_left(ex_list[k][i], LOs[k][i], A[i])
            ROs[k][Ns] = np.ones((1, 1))
            for i in range(Ns - 1, -1, -1):
                ROs[k][i] = _overlap_right(ex_list[k][i], ROs[k][i + 1], A[i])
        for i in range(0, Ns - 1):
            Th = np.tensordot(A[i], A[i + 1], axes=([2], [0]))
            op = (lambda v, Li=L[i], Ri=Rset[i + 2], W1=mpo[i], W2=mpo[i + 1]:
                  apply_heff(Li, W1, W2, Ri, v))
            Th0 = Th
            for k in range(nex):
                ph = _overlap_vec(ex_list[k], LOs[k][i], ROs[k], i)
                op = _add_penalty(op, ph, lam)
                if lam is None:
                    nrm = float(np.real(np.vdot(ph, ph)))
                    if nrm > 1e-30:
                        Th0 = Th0 - ph * (np.vdot(ph, Th0) / nrm)
            e, vec, used = lanczos(op, Th0, iters=iters)
            A[i], A[i + 1] = _split(vec, Th.shape[0], Th.shape[3], chi, right_side=False)
            L[i + 1] = _update_left(L[i], mpo[i], A[i])
            for k in range(nex):
                LOs[k][i + 1] = _overlap_left(ex_list[k][i], LOs[k][i], A[i])
        Rset = right_envs(mpo, A)
        for i in range(Ns - 2, -1, -1):
            Th = np.tensordot(A[i], A[i + 1], axes=([2], [0]))
            op = (lambda v, Li=L[i], Ri=Rset[i + 2], W1=mpo[i], W2=mpo[i + 1]:
                  apply_heff(Li, W1, W2, Ri, v))
            Th0 = Th
            for k in range(nex):
                ph = _overlap_vec(ex_list[k], LOs[k][i], ROs[k], i)
                op = _add_penalty(op, ph, lam)
                if lam is None:
                    nrm = float(np.real(np.vdot(ph, ph)))
                    if nrm > 1e-30:
                        Th0 = Th0 - ph * (np.vdot(ph, Th0) / nrm)
            e, vec, used = lanczos(op, Th0, iters=iters)
            A[i], A[i + 1] = _split(vec, Th.shape[0], Th.shape[3], chi, right_side=True)
            Rset[i + 1] = _update_right(mpo[i + 1], A[i + 1], Rset[i + 2])
            for k in range(nex):
                ROs[k][i + 1] = _overlap_right(ex_list[k][i + 1], ROs[k][i + 2], A[i + 1])
        E = energy(mpo, A)
        hist.append(E)
        if verbose:
            print("   sweep %d: E=%.12f chi=%d" % (sw, E, max(a.shape[2] for a in A)))
        if sw > 1 and abs(hist[-1] - hist[-2]) < tol:
            break
    return {"A": A, "E": E, "hist": hist, "chi": max(a.shape[2] for a in A)}

def efield_mpo(N: int, xcut: int):
    """E_x = Σ_{y≤x}ρ_y 的 2 通道 MPO（做激发态初态用）。"""
    mpo = []
    for x in range(N):
        W = np.zeros((2, 2, 2, 2))
        W[0, :, :, 0] += _I2
        W[1, :, :, 1] += _I2
        if x <= xcut:
            W[0, :, :, 1] += _rho_op(x)
        mpo.append(W)
    return mpo


def apply_mpo(Ws, A):
    """把 MPO 作用到 MPS（不压缩，键维 ×D），边界通道抽出。"""
    N = len(A)
    D = Ws[0].shape[0]
    B = []
    for i in range(N):
        T = np.tensordot(Ws[i], A[i], axes=([2], [1]))
        B.append(T.transpose(3, 0, 1, 4, 2).reshape(A[i].shape[0] * D, 2, A[i].shape[2] * D))
    B[0] = B[0][0:1]
    B[-1] = B[-1][:, :, -1:]
    return B


def compress(A, chi, sweeps: int = 4):
    """把（可能超大的）MPS 压回键维 chi。"""
    A = right_canonical(A)
    N = len(A)
    for _ in range(sweeps):
        for i in range(N - 1):
            l, d, r = A[i].shape
            U, S, Vt = np.linalg.svd(A[i].reshape(l * d, r), full_matrices=False)
            k = min(chi, S.size)
            A[i] = U[:, :k].reshape(l, d, k)
            A[i + 1] = np.tensordot((S[:k, None] * Vt[:k]), A[i + 1], axes=([1], [0]))
        for i in range(N - 1, 0, -1):
            l, d, r = A[i].shape
            U, S, Vt = np.linalg.svd(A[i].reshape(l, d * r), full_matrices=False)
            k = min(chi, S.size)
            A[i] = Vt[:k].reshape(k, d, r)
            A[i - 1] = np.tensordot(A[i - 1], (U[:, :k] * S[:k]).T, axes=([2], [1]))
    return A


def dmrg_gap(N: int, g: float, m: float = 0.0, chi: int = 64, sweeps: int = 12,
             t: float = 0.5, seed: int = 5, verbose: bool = False, phi: float = 0.0):
    """基态 + 第一激发态（正交 DMRG），返回 E0/E1/gap/gap_over_e。

    `phi` 原样传给 `mpo_ks`（磁通）。φ ≠ 0 时 H 是复厄米，DMRG 走复数分支。
    """
    r0 = dmrg_ks(N, g, m, chi=chi, sweeps=sweeps, t=t, seed=seed, verbose=verbose,
                 phi=phi)
    B = compress(apply_mpo(efield_mpo(N, N // 2), right_canonical(r0["A"])), chi)
    r1 = dmrg_ks(N, g, m, chi=chi, sweeps=sweeps + 2, t=t, seed=seed + 4,
                 A_init=B, excited=r0["A"], lam=None, tol=0.0, iters=40,
                 verbose=verbose, phi=phi)
    e = g * np.sqrt(2.0 * t)
    return {"N": N, "g": g, "t": t, "e": float(e), "x": float(e / (2.0 * t)),
            "E0": float(r0["E"]), "E1": float(r1["E"]),
            "gap": float(r1["E"] - r0["E"]),
            "gap_over_e": float((r1["E"] - r0["E"]) / e) if e > 0 else float("nan"),
            "A": r0["A"], "A1": r1["A"],
            "bond": int(max(a.shape[2] for a in r0["A"])),
            "sweeps": (len(r0["hist"]), len(r1["hist"]))}


def dmrg_multi_gap(N: int, g: float, j_max: int = 4, m: float = 0.0,
                   chi: int = 64, sweeps: int = 12, t: float = 0.5,
                   seed: int = 5, verbose: bool = False, phi: float = 0.0):
    """基态 + j_max 个激发态（正交 DMRG），按能量升序返回。

    每个激发态用 E_xcut|0⟩（xcut 逐个轮换）做初态，然后投影掉**所有**低态。
    用于色散关系测量：需要 j = 0..4（即 5 个独立动量值）时，在 φ = 0 上
    最多出 9 个激发态（k = 0 不简并 + 4 对 ±k 简并）。
    返回 {states: [A0, A1, ...], energies: [E0, E1, ...], e, gaps}。
    """
    r0 = dmrg_ks(N, g, m, chi=chi, sweeps=sweeps, t=t, seed=seed,
                 verbose=verbose, phi=phi)
    states  = [r0["A"]]
    energies = [r0["E"]]
    e = g * np.sqrt(2.0 * t)
    for j in range(j_max):
        xcut = (N // 2 + j * N // max(j_max, 1)) % N
        B = compress(apply_mpo(efield_mpo(N, xcut),
                               right_canonical(states[0])), chi)
        rj = dmrg_ks(N, g, m, chi=chi, sweeps=sweeps + 2, t=t,
                     seed=seed + 4 * (j + 1), A_init=B, excited=states,
                     lam=None, tol=0.0, iters=40, verbose=verbose, phi=phi)
        states.append(rj["A"])
        energies.append(rj["E"])
    gaps = [float(ej - energies[0]) for ej in energies[1:]]
    return {"N": N, "g": g, "t": t, "e": float(e),
            "x": float(e / (2.0 * t)) if t > 0 else float("nan"),
            "states": states, "energies": energies,
            "E0": float(energies[0]), "gaps": gaps,
            "bond": int(max(a.shape[2] for a in states[0]))}



# ==============================================================================
# 5. 关联函数与"第二个 m_γ"
# ==============================================================================
def corr_rho(A, ops):
    """C[r] = ⟨ρ_r ρ_0⟩（固定 0 号点，一次左扫 O(Nχ³)）。A 需归一。"""
    N = len(A)
    R = [None] * (N + 1)
    R[N] = np.ones((1, 1))
    for i in range(N - 1, -1, -1):
        X = np.tensordot(A[i], R[i + 1], axes=([2], [0]))
        R[i] = np.tensordot(X, A[i].conj(), axes=([1, 2], [1, 2]))
    norm = R[0][0, 0]
    V = np.tensordot(np.tensordot(A[0][0], ops[0], axes=([0], [0])),
                     A[0][0].conj(), axes=([1], [0]))
    out = [float(np.sum(V * R[1]))]
    for r in range(1, N):
        W1 = np.tensordot(V, A[r], axes=([0], [0]))            # (B,s,g)
        W2 = np.tensordot(W1, ops[r], axes=([1], [1]))         # (B,g,S)
        W3 = np.tensordot(W2, A[r].conj(), axes=([0, 2], [0, 1]))
        out.append(float(np.sum(W3 * R[r + 1])))
        V = np.tensordot(W1, A[r].conj(), axes=([0, 1], [0, 1]))
    return np.array(out) / norm


def corr_rho_from(A, ops, origin):
    """C[d] = ⟨ρ_{origin+d} ρ_origin⟩，d = 0..N−1−origin（一次左→右扫，O(Nχ³)）。

    与 `corr_rho` 的区别：起点可以不在 0 号格点。环上基态被切口破坏平移不变
    （交错费米子的 CDW 在接缝处有缺陷），所以 m_γ 的关联必须按**所有起点**取，
    不能假设 C 只依赖 (起点宇称, 距离)。
    """
    N = len(A)
    R = [None] * (N + 1)
    R[N] = np.ones((1, 1))
    for i in range(N - 1, -1, -1):
        X = np.tensordot(A[i], R[i + 1], axes=([2], [0]))
        R[i] = np.tensordot(X, A[i].conj(), axes=([1, 2], [1, 2]))
    L = [None] * (N + 1)
    L[0] = np.ones((1, 1))
    for i in range(N):
        X = np.tensordot(L[i], A[i], axes=([0], [0]))
        L[i + 1] = np.tensordot(X, A[i].conj(), axes=([0, 1], [0, 1]))
    norm = float(np.real(L[N][0, 0]))
    y = int(origin)

    def _seed(op):
        v = np.tensordot(np.tensordot(L[y], A[y], axes=([0], [0])), op, axes=([1], [1]))
        return np.tensordot(v, A[y].conj(), axes=([0, 2], [0, 1]))      # (B, g)

    V = _seed(ops[y])                              # 向右传播用它（同点只有一个算符）
    out = [float(np.real(np.sum(_seed(ops[y] @ ops[y]) * R[y + 1]))) / norm]
    for r in range(y + 1, N):
        W1 = np.tensordot(V, A[r], axes=([0], [0]))
        W2 = np.tensordot(W1, ops[r], axes=([1], [1]))
        W3 = np.tensordot(W2, A[r].conj(), axes=([0, 2], [0, 1]))
        out.append(float(np.real(np.sum(W3 * R[r + 1]))) / norm)
        V = np.tensordot(W1, A[r].conj(), axes=([0, 1], [0, 1]))
    return np.array(out)


def one_site_all(A, ops):
    """⟨op_x⟩ 逐点（一次左扫）。"""
    N = len(A)
    R = [None] * (N + 1)
    R[N] = np.ones((1, 1))
    for i in range(N - 1, -1, -1):
        X = np.tensordot(A[i], R[i + 1], axes=([2], [0]))
        R[i] = np.tensordot(X, A[i].conj(), axes=([1, 2], [1, 2]))
    norm = R[0][0, 0]
    out = np.zeros(N)
    V = np.ones((1, 1))
    for i in range(N):
        W1 = np.tensordot(V, A[i], axes=([0], [0]))
        W2 = np.tensordot(W1, ops[i], axes=([1], [1]))
        W3 = np.tensordot(W2, A[i].conj(), axes=([0, 2], [0, 1]))
        out[i] = float(np.sum(W3 * R[i + 1])) / norm
        V = np.tensordot(W1, A[i].conj(), axes=([0, 1], [0, 1]))
    return out


def corr_electric(C, rho, N: int, h: int = 2):
    """⟨Ẽ_x Ẽ_0⟩_conn，Ẽ_x = (1/h)Σ_{y=x}^{x+h−1}E_y（默认 h=2 抹平交错分量）。

    输入 C[r] = ⟨ρ_rρ_0⟩、rho[x] = ⟨ρ_x⟩；返回长度 N−h 的数组（x = 0..N−h−1）。
    """
    EE = np.cumsum(C)
    Em = np.cumsum(rho)

    def seg(a, k):
        return sum(a[k + j] for j in range(h) if k + j < N)

    out = np.zeros(N - h)
    v0, m0 = seg(EE, 0), seg(Em, 0)
    for x in range(N - h):
        out[x] = (seg(EE, x) - seg(Em, x) * m0) / h / h
    return out


def xi_from_corr(y, N: int, h: int = 2, fit_frac=(0.15, 0.55)):
    """从关联函数拟合衰减长度 ξ：y(x) = A(e^{−x/ξ} + e^{−(N−x)/ξ})。

    只在 N·fit_frac 的窗口内拟合（避开 x≈0 的短程污染与 x≈N/2 的绕行像）。
    """
    y = np.asarray(y, dtype=float)
    lo = max(1, int(fit_frac[0] * N))
    hi = max(lo + 3, int(fit_frac[1] * N))
    xs = np.arange(lo, min(hi, len(y)))
    ys = y[xs]
    best = None
    for xi in np.exp(np.linspace(np.log(0.5), np.log(0.5 * N), 4000)):
        f = np.exp(-xs / xi) + np.exp(-(N - xs) / xi)
        A = float(np.dot(f, ys) / np.dot(f, f))
        dev = float(np.max(np.abs(A * f - ys)) / max(abs(A), 1e-300))
        if best is None or dev < best[2]:
            best = (A, xi, dev)
    return {"A": best[0], "xi": float(best[1]), "m": float(1.0 / best[1]),
            "dev": best[2], "window": (int(xs[0]), int(xs[-1]))}


# ==============================================================================
# 6. 连续极限：实测网格与外推
# ==============================================================================
# (x, N, gap/e)：χ = 96、pen = 1、无电场截断的两格点 DMRG 实测值。
# 复现：仓库根 `python ks_mps_scan.py`（单点 N=60 约 140 s，全表约 40 min）。
MPS_GAP_SCAN = (
    (0.15, 16, 0.142244),
    (0.15, 20, 0.161081),
    (0.15, 24, 0.180910),
    (0.15, 32, 0.223592),
    (0.15, 40, 0.269334),
    (0.15, 48, 0.316615),
    (0.15, 60, 0.386261),
    (0.20, 16, 0.199296),
    (0.20, 20, 0.230061),
    (0.20, 24, 0.262278),
    (0.20, 32, 0.329106),
    (0.20, 40, 0.395082),
    (0.20, 48, 0.455139),
    (0.20, 60, 0.525157),
    (0.30, 16, 0.323430),
    (0.30, 20, 0.378164),
    (0.30, 24, 0.430963),
    (0.30, 32, 0.521366),
    (0.30, 40, 0.581443),
    (0.30, 48, 0.612513),
    (0.30, 60, 0.628777),
    (0.40, 16, 0.447283),
    (0.40, 20, 0.514664),
    (0.40, 24, 0.568882),
    (0.40, 32, 0.631774),
    (0.40, 40, 0.652440),
    (0.40, 48, 0.656997),
    (0.40, 60, 0.656921),
    (0.50, 16, 0.555954),
    (0.50, 20, 0.617261),
    (0.50, 24, 0.653496),
    (0.50, 32, 0.678215),
    (0.50, 40, 0.680745),
    (0.50, 48, 0.679613),
    (0.50, 60, 0.677779),
)

# (N, x, chi, gap/e)：χ 收敛表（同一批参数只改 χ）。
MPS_CHI_SCAN = (
    (32, 0.20, 48, 0.329458),
    (32, 0.20, 96, 0.329106),
    (32, 0.20, 128, 0.329099),
    (40, 0.15, 48, 0.270294),
    (40, 0.15, 96, 0.269334),
    (40, 0.15, 128, 0.269341),
)


def _per_x_rows(points, n_min):
    """每个 x：最大 N 的实测值 + 与次大 N 的漂移（漂移就是没收敛的量度）。"""
    groups = {}
    for x, n, v in points:
        if n >= n_min:
            groups.setdefault(round(float(x), 9), []).append((int(n), float(v)))
    rows = []
    for x in sorted(groups):
        g = sorted(groups[x])
        if len(g) < 2:
            rows.append({"x": x, "G": g[-1][1], "drift": float("nan"),
                         "N": [g[-1][0], g[-1][0]]})
            continue
        rows.append({"x": x, "G": g[-1][1], "drift": abs(g[-1][1] - g[-2][1]),
                     "N": [g[-2][0], g[-1][0]]})
    return rows


def fit_continuum_gap_mps(points=None, n_min: int = 16, max_drift: float = 0.02,
                          chi_dev: float = 0.0):
    """把 gap/e 外推到 x→0，返回 G(0) 与不确定度。

    **两步，都用实测的数，不预设答案**：

    ① 固定 x：取**最大 N** 的实测值当 G(x)，并报它与次大 N 的差 `drift`。
       不做指数外插——实测表明 x ≤ 0.3 的序列在 N ≤ 60 内根本不饱和
       （见 OPERATOR_STATUS.md §6quaterdecies 的漂移表），硬套 e^{−κN} 会把
       G(x) 外推到 6.0 这种非物理值。`drift` 就是这一项的诚实误差。
    ② 只在 `drift ≤ max_drift` 的 x（即真的收敛了的 x）上做 `G(x) = G0 + A·x^p`，
       p 由网格搜索定；不确定度 = 各拟合形式的极差、被用掉的 drift、χ 截断偏差。

    这样 G(0) 只用"看得见收敛"的点，代价是 x 范围窄、不确定度大——这是数据的样子，
    不是拟合者的选择。
    """
    pts = MPS_GAP_SCAN if points is None else points
    pts = [(float(a), int(b), float(c)) for a, b, c in pts]
    rows = _per_x_rows(pts, n_min)
    use = [r for r in rows if np.isfinite(r["drift"]) and r["drift"] <= max_drift]
    out = {"target": SQRT_PI_INV, "rows": rows, "usable": [r["x"] for r in use],
           "max_drift": max_drift, "n_points": len(pts), "forms": [],
           "x_range": [float(rows[0]["x"]), float(rows[-1]["x"])] if rows else [0.0, 0.0],
           "N_range": [min(r["N"][1] for r in rows), max(r["N"][1] for r in rows)] if rows else [0, 0]}
    if len(use) < 2:
        out.update({"G0": float("nan"), "dev_rel": float("nan"), "spread": float("nan"),
                    "unc": float("nan"), "p": float("nan"), "drift_used": float("nan")})
        return out
    xv = np.array([r["x"] for r in use])
    gv = np.array([r["G"] for r in use])
    drift = float(max(r["drift"] for r in use))

    def power_fit(p):
        A = np.vstack([np.ones_like(xv), xv ** p]).T
        c, *_ = np.linalg.lstsq(A, gv, rcond=None)
        return float(c[0]), float(np.max(np.abs(A @ c - gv)))

    best = None
    for p in np.linspace(0.3, 3.0, 271):
        g0, dev = power_fit(float(p))
        if best is None or dev < best[2]:
            best = (g0, float(p), dev)
    g0_free, p_hat, dev_free = best
    forms = []
    for pf in (1.0, 1.5, 2.0):
        gg, dd = power_fit(pf)
        forms.append(("p=%.1f" % pf, gg, dd))
    if len(xv) >= 3:
        A = np.vstack([np.ones_like(xv), xv ** 2, xv ** 4]).T
        c, *_ = np.linalg.lstsq(A, gv, rcond=None)
        forms.append(("c1x²+c2x⁴", float(c[0]),
                      float(np.max(np.abs(A @ c - gv)))))
    # 只有 3 个可用 x 点时，p 的自由拟合和 3 参数多项式都是"刚好凑上"的（残差恒为 0），
    # 所以中心值取固定 p 家族的**均值**，不确定度取家族极差——不假装 p 能定出来。
    g0s = [g for _, g, _ in forms]
    g0 = float(np.mean(g0s))
    dev0 = max(d for _, _, d in forms)
    spread = max(g0s) - min(g0s)
    out.update({"G0": g0, "p": p_hat, "dev_rel": abs(g0 - SQRT_PI_INV) / SQRT_PI_INV,
                "spread": spread, "drift_used": drift,
                "unc": float(np.hypot(np.hypot(spread, drift), chi_dev)),
                "max_dev_x": dev0, "forms": [(n, g) for n, g, _ in forms],
                "p_free": p_hat, "G0_free": g0_free, "dev_free": dev_free})
    return out


def _dense_module():
    try:
        from . import ks_ed
        return ks_ed
    except ImportError:                                   # pragma: no cover
        import ks_ed
        return ks_ed


def _lanczos_module():
    try:
        from . import ks_lanczos
        return ks_lanczos
    except ImportError:                                   # pragma: no cover
        import ks_lanczos
        return ks_lanczos


def _q2_mpo(N: int):
    """(Σρ)² 的 3 通道 MPO：用来量电荷扇区是否被钉住。"""
    out = []
    for x in range(N):
        W = np.zeros((3, 2, 2, 3))
        rho = _rho_op(x)
        W[0, :, :, 0] += _I2
        W[0, :, :, 1] += rho
        W[1, :, :, 1] += _I2
        W[1, :, :, 2] += 2.0 * rho          # 对 (y<z) 的权重是 2ρ_yρ_z
        W[0, :, :, 2] += rho @ rho
        W[2, :, :, 2] += _I2
        out.append(W)
    return out


def _xi_synthetic(target=8.0, N=64):
    """在合成指数上验证 xi_from_corr：y(x) = e^{−x/ξ} + e^{−(N−x)/ξ}。"""
    x = np.arange(N - 2, dtype=float)
    y = np.exp(-x / target) + np.exp(-(N - x) / target)
    return abs(xi_from_corr(y, N, fit_frac=(0.10, 0.50))["xi"] - target) / target


def _ed_mps_mismatch(N, g, chi, k=2):
    """ED（Emax 饱和）与 DMRG 的 E0/gap/关联函数逐项对比。"""
    ked = _dense_module()
    Emax = N + 2                       # 饱和：不存在 |E| 超过它的 Q=0 构型
    H, masks, _ = ked.basis_and_hamiltonian(N, g, 0.0, t=0.5, Emax=Emax)
    w, v = np.linalg.eigh(H)
    r = dmrg_gap(N, g, chi=chi, sweeps=10, seed=5)
    ops = [_rho_op(x) for x in range(N)]
    C = corr_rho(r["A"], ops)
    idx = np.arange(1 << N)
    bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
    prob = np.zeros(1 << N)
    prob[masks] = v[:, 0] ** 2
    C_ed = np.array([float(np.sum(prob * _rho_op(x)[bits[:, x], bits[:, x]]
                                  * _rho_op(0)[bits[:, 0], bits[:, 0]])) for x in range(N)])
    rho_mps = one_site_all(r["A"], ops)
    rho_ed = np.array([float(np.sum(prob * _rho_op(x)[bits[:, x], bits[:, x]]))
                       for x in range(N)])
    return {"E0": abs(r["E0"] - w[0]), "gap": abs(r["gap"] - (w[1] - w[0])),
            "corr": float(np.abs(C - C_ed).max()),
            "rho": float(np.abs(rho_mps - rho_ed).max()), "r": r, "ed": w}


def check_ks_mps() -> dict:
    """P36：Kogut–Susskind 质量隙的连续极限（MPS/DMRG，N 到 60、x 到 0.15）。

    P35 用矩阵自由 Lanczos 推到 N = 24，三重外推给 G(0) = 0.594 ± 0.073（比 1/√π
    高 5.2%）；要判定这 5% 需要 N ≈ 57，稠密基矢 4^57 不可能。本项换成 MPS/DMRG：
    代价随 N 线性、随键维三次。

    **结构判据（进残差，全部应为机器精度或明确阈值）**
      ① MPO vs `ks_ed` 的稠密 H：同一个 H，逐元素一致（N=6/8，m=0 与 m≠0）
      ② DMRG vs 稠密 ED：N=8/10/12/14 的 E0 逐个数字对上（≤1e-10）
      ③ 同一个 gap（≤1e-10）
      ④ ⟨ρ_rρ_0⟩ 与 ED 的差（≤1e-8）
      ⑤ χ 收敛：N=16、x=0.5 的 gap 在 χ=32→64 的变化 ≤ 1e-4 相对
      ⑥ 与 `ks_lanczos`（独立实现，Emax=2）在 N=16 的一致性 ≤ 1e-3 相对
         ——差的正是 Emax 截断本身的量级，说明"MPS 侧不对电场截断"没引入新误差
      ⑦ 电荷扇区：⟨Q²⟩ ≤ 1e-18（pen·Q² 罚项真的把态钉在 Σρ = 0）
      ⑧ 表内两点现场复现（残差 ≤ 1e-6）：证明 MPS_GAP_SCAN 不是抄来的
    **物理结论（_magnitude，只报数不判定）**
      · 连续极限 G(0)、与 1/√π 的偏差、拟合形式散布、格点伪影幂次 p
      · 第二观测量：⟨E(x)E(0)⟩_conn 的衰减长度给的 m_γ
    """
    out = {}
    ked = _dense_module()

    # ① MPO vs 稠密 H（限制到 Q=0 子空间，与 ks_ed 同一基）
    worst = 0.0
    for N in (6, 8):
        for (g, m) in ((1.0, 0.0), (0.7, 0.4)):
            H_ed, masks, _ = ked.basis_and_hamiltonian(N, g, m, t=0.5, Emax=N + 2)
            mpo, D = mpo_ks(N, g, m, t=0.5)
            H_mpo = mpo_dense(mpo, D)
            worst = max(worst, float(np.abs(H_mpo[np.ix_(masks, masks)] - H_ed).max()))
    out["MPO vs 稠密 H（N=6/8, m=0/0.4）"] = worst

    # ②③④ DMRG vs 稠密 ED
    e0w = gapw = corrw = rhow = 0.0
    for (N, g, chi) in ((8, 1.0, 64), (10, 0.7, 64), (12, 0.5, 96), (14, 1.0, 96)):
        d = _ed_mps_mismatch(N, g, chi)
        e0w = max(e0w, d["E0"])
        gapw = max(gapw, d["gap"])
        corrw = max(corrw, d["corr"])
        rhow = max(rhow, d["rho"])
    out["DMRG vs 稠密 ED：E0（N=8/10/12/14）"] = e0w
    out["DMRG vs 稠密 ED：gap（N=8/10/12/14）"] = gapw
    out["DMRG vs 稠密 ED：⟨ρ_rρ_0⟩（同上）"] = corrw
    out["DMRG vs 稠密 ED：⟨ρ_x⟩ 逐点（同上）"] = rhow
    out["xi_from_corr 在合成指数上复原 ξ"] = _xi_synthetic()

    # ⑤ χ 收敛（N=16, x=0.5）
    a = dmrg_gap(16, 0.5, chi=32, sweeps=10, seed=3)
    b = dmrg_gap(16, 0.5, chi=64, sweeps=10, seed=3)
    chi_dev = abs(b["gap_over_e"] - a["gap_over_e"]) / abs(b["gap_over_e"])
    out["χ 收敛 Δgap(64←32)/gap（N=16,x=0.5，判据 ≤1e-4）"] = max(0.0, chi_dev - 1e-4)
    out["χ 收敛 Δgap(64←32)/gap_magnitude"] = chi_dev

    # ⑥ 与 ks_lanczos（Emax=2 的独立实现）在 N=16、x=0.5 的对照
    kl = _lanczos_module()
    ref = kl.gap_lanczos(16, 0.5, t=0.5, Emax=2, iters=200)
    dev06 = abs(b["gap"] - ref["gap"]) / ref["gap"]
    out["vs ks_lanczos(Emax=2) Δgap/gap（N=16,x=0.5，判据 ≤1e-3）"] = max(0.0, dev06 - 1e-3)
    out["vs ks_lanczos(Emax=2) Δgap/gap_magnitude"] = dev06

    # ⑦ 电荷扇区
    mpo, D = mpo_ks(16, 0.5, 0.0, t=0.5, pen=0.0)
    q2 = _q2_mpo(16)
    out["电荷扇区 ⟨Q²⟩（N=16,x=0.5）"] = abs(energy(q2, b["A"]))

    # ⑧ 表内两点现场复现
    worst = 0.0
    hit = 0
    for (x, N) in ((0.50, 16), (0.30, 16)):
        tab = [v for (xx, nn, v) in MPS_GAP_SCAN if abs(xx - x) < 1e-9 and nn == N]
        if not tab:
            continue
        hit += 1
        g = dmrg_gap(N, x, chi=64, sweeps=10, seed=3)
        worst = max(worst, abs(g["gap_over_e"] - tab[0]))
    out["表内两点现场复现"] = worst if hit else 1.0

    # 物理结论
    fit = fit_continuum_gap_mps(chi_dev=chi_dev)
    out["连续极限 G(0) = gap/e_magnitude"] = fit["G0"]
    out["1/√π_magnitude"] = fit["target"]
    out["|G(0) − 1/√π|/(1/√π)_magnitude"] = fit["dev_rel"]
    out["外推不确定度_magnitude"] = fit["unc"]
    out["拟合形式散布_magnitude"] = fit["spread"]
    out["p 自由拟合（3 点退化，仅供参考）_magnitude"] = fit.get("p_free", float("nan"))
    out["可用于 x→0 外推的 x_magnitude"] = float(len(fit["usable"]))
    out["最大 N 值的漂移（被用到的点）_magnitude"] = fit.get("drift_used", float("nan"))
    for r in fit["rows"]:
        out["gap/e（x=%.2f, 最大 N）_magnitude" % r["x"]] = r["G"]
        out["gap/e 漂移（x=%.2f）_magnitude" % r["x"]] = r["drift"]
    if len(MPS_CHI_SCAN):
        out["χ 表最大 bond_magnitude"] = float(max(r[2] for r in MPS_CHI_SCAN))
    return out
