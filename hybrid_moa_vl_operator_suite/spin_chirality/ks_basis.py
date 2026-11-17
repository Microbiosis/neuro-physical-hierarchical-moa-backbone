# -*- coding: utf-8 -*-
"""1+1D Kogut–Susskind 的**平移协变基**：状态显式携带电场 `(mask, E)`，平移就是纯重标号。

模型（环、N 个格点、交错费米子、U(1) 电场、a = 1）
----------------------------------------------------
    H = (g²/2) Σ_x E_x²  −  t Σ_x (χ†_x U_x χ_{x+1} + h.c.)  +  m Σ_x (−1)^x n_x

    格点   x = 0…N−1（N 偶）；E_x 是 x 与 x+1 之间那根链接上的电场（x = N−1 绕回来）
    电荷   ρ_x = n_x − ½(1 − (−1)^x) ∈ {0, ±1}（整数，所以 E 也是整数）
    Gauss  E_x − E_{x−1} = ρ_x 逐态成立；环上求和给出 Σρ = 0（半满：Σn = N/2）
    截断   |E_x| ≤ Emax；跳跃越界就整对丢掉（`build_h`），厄米性不受影响
    跳跃   χ†_a U_x χ_b 只改**链接 x** 上的那一个分量：E_x → E_x ± 1（a = x 时抬）

基
--
    |mask, E⟩ ：E 是**独立的整数向量**，不再由「常数 c + 前缀和」折出来。

旧版（`ks_ed.py` 与 `ks_basis.py` 前几轮）用 `(mask, c)`、E = c + Σ_{y≤x}ρ_y。环上这是有代价的：
前缀和在接缝处断了一次（c 就是接缝 N−1 上的电场），于是

  · 平移会挪接缝：E → roll(E, 2) 在标签里是 c' = c − (ρ_{N−2} + ρ_{N−1})，**不是** c 不变；
    旧代码靠 `round(E'[0] − cumsum(ρ')[0])` 反算，一处写错就全错；
  · 绕圈那根键（链接 N−1）的跳跃要 c → c ± 1，普通键 c 不变——两类键两套规则，
    就是那批错配的来源——材料里的中间版记的是 73 条，HEAD 实测 259 条（而且它在截断上也不对称，见下）。

E 存成显式向量后这两件事都自动对：跳跃只改链接 x 的一个分量，平移是 (mask, E) 的 roll，
没有任何 c 要算。**绕圈键与普通键从此完全同构**，这就是 [H,T] = 0 的全部内容。

平移是 2 格，不是 1 格
---------------------
交错费米子按 x 的**绝对奇偶**定义 ρ，1 格平移会把交错结构翻过来
（ρ'_x = ρ_{x−1} ∓ 1，不再是 ρ_{x−1}），Gauss 定律整体破坏。实测 N=8、Emax=2 的
**257/257** 个态，1 格像全都不在基里。物理的平移单位是**一个单位胞（2 格）**，T 的阶是 N/2。

绕圈键的 JW 串：是约定，不是物理
--------------------------------
标准一维 Jordan–Wigner（排序 0<1<…<N−1）只给绕圈键一个态依赖的符号
`(−1)^{N_f − n_0 − n_{N−1}}`（其余键的串都是 1）。它在 2 格平移下不是不变量，实测
‖[H,T]‖ = 1.0、292 条矩阵元对不上；去掉它（每根键幅度都是 −t）以后 ‖[H,T]‖ = 0。
这不是改物理：两个 H 的谱**逐点相同**（实测 max|ΔE| = 5.0e-14），差别只是一个对角 ±1 的
态重定义（JW 串本身），也就是边界条件的约定。Susskind 的自旋-1/2 表示里这点是显然的：
偶格点 = 物质 σ^z、奇格点 = 规范 τ^z、跳跃 = σ⁺τ⁺σ⁻，**没有 JW 串**，绕圈键与普通键一样。

旧代码在 HEAD 上还有第二个真 bug：跳跃时把电场往「a < b 的那一侧」挪，而不是往**链接 x**
挪；绕圈键上 a = N−1 > b = 0，于是方向反了，检查 |E ± 1| ≤ Emax 时把**合法**的跳跃丢掉、
又留下了反向的那一半 ⇒ ‖H − Hᵀ‖ = 0.5（66 对不对称）。现在的构造里方向只由链接决定，
截断只由目标态是否在基里决定，成对丢掉 ⇒ 厄米是恒等式。

实测（N = 8、Emax = 2、dim = 257）
---------------------------------
  · ‖H − Hᵀ‖ = 0；Gauss 定律逐态残差 0；
  · ‖[H,T]‖ = 0（g = 0.2 / 0.4 / 1.0 都是**精确** 0，不是 1e-16 量级）；
  · T 是置换 ⇒ 幺正；T^{N/2} = T^N = 1；
  · 按 T 的轨道做 Fourier 变换后 H **严格分块对角**（跨扇区最大矩阵元 1.4e-16，纯浮点）；
    扇区标签 p = m/L，格点动量 k = πp（k 与 k+π 共用一个 T 本征值，故折叠在 [0, π)）。
  · 换尺寸复核 N = 10（dim = 874）：`[H,T]`、矩阵元错配、`T^N = 1` 同样为 0，分块对角 3.6e-15。

状态（2026-11-13）：P41 已接入审计（`operator._check_ks_basis`），判据全部进残差。
"""
from __future__ import annotations

import numpy as np

__all__ = ["enumerate_basis", "build_h", "translation", "momentum_sectors", "check_basis"]


def _rho(N: int, mask: int) -> tuple:
    """ρ_x = n_x − ½(1 − (−1)^x)：偶格点 ρ = n，奇格点 ρ = n − 1，取值都是整数。"""
    return tuple(((mask >> x) & 1) - (0 if x % 2 == 0 else 1) for x in range(N))


def _rot(mask: int, N: int, step: int) -> int:
    """把占据从格点 x 搬到 x + step（环上）。"""
    step %= N
    return sum((((mask >> x) & 1) << ((x + step) % N)) for x in range(N))


def _shift(N: int, mask: int, E: tuple, step: int):
    """平移一个单位：占据搬到 x+step，链接上的电场跟着搬（E'_x = E_{x−step}）。"""
    return _rot(mask, N, step), tuple(E[(x - step) % N] for x in range(N))


def enumerate_basis(N: int, Emax: int = 2):
    """枚举物理基：{(mask, E)}，E 是整数元组，逐点满足 Gauss 定律且 |E_x| ≤ Emax。

    返回 `(states, index)`，`states[i] = (mask, E)`；`index` 是反查表
    `{(mask, E): i}`。同一个 mask 的各扇区（不同的「真空角」）是**不同的态**，
    所以 E 必须显式存：给定 mask，E 由接缝上的值决定，而接缝本身会随平移移动。

    环上 Gauss 定律求和给出 Σρ = 0，即 Σn = N/2 —— 基只在半满扇区里，这是构造使然，
    不是额外截断。
    """
    states = []
    for mask in range(1 << N):
        r = _rho(N, mask)
        if sum(r) != 0:                       # 环上总电荷必须为零
            continue
        cum = np.cumsum(r)
        lo = int(np.ceil(-Emax - cum.min()))  # 使 max|E| ≤ Emax 的全部「真空角」
        hi = int(np.floor(Emax - cum.max()))
        for c in range(lo, hi + 1):
            states.append((mask, tuple(int(v) for v in (cum + c))))
    return states, {s: i for i, s in enumerate(states)}


def _jw_wrap_sign(mask: int, N: int) -> float:
    """标准一维 JW 排序给绕圈键的符号 (−1)^{N_f − n_0 − n_{N−1}}；其余键的串都是 1。

    只用于 `build_h(boundary="jw")` 的对照：它让 H 失去 2 格平移对称性（但谱不变）。
    """
    return -1.0 if (bin(mask).count("1") - ((mask >> 0) & 1) - ((mask >> (N - 1)) & 1)) % 2 else 1.0


def build_h(N: int, g: float, m: float = 0.0, t: float = 0.5, Emax: int = 2,
            boundary: str = "periodic"):
    """稠密厄米 H 与基。返回 `(H, states, index)`。

    跳跃项写成**成对加项**：对每个态、每根键、两个方向都写一次，任一端越界就整对丢掉
    （目标态不在基里 ⇒ 该项为 0，反向也从「不存在的态」出发 ⇒ 两边同时为 0）。
    因此厄米性是恒等式，不是靠调出来的。

    `boundary="periodic"`（默认）每根键幅度都是 −t，H 与 2 格平移对易；
    `boundary="jw"` 只给绕圈键补上标准 JW 串的符号（旧写法），谱相同但与平移**不**对易，
    留作对照（`check_basis` 里同时给出两个数）。
    """
    if boundary not in ("periodic", "jw"):
        raise ValueError(f"boundary 只能是 'periodic' / 'jw'，收到 {boundary!r}")
    states, index = enumerate_basis(N, Emax)
    dim = len(states)
    H = np.zeros((dim, dim))
    for i, (mask, E) in enumerate(states):
        diag = 0.5 * g * g * float(sum(e * e for e in E))
        diag += m * sum((1 if x % 2 == 0 else -1) * ((mask >> x) & 1) for x in range(N))
        H[i, i] = diag
    for i, (mask, E) in enumerate(states):
        for x in range(N):
            y = (x + 1) % N
            for a, b in ((x, y), (y, x)):           # χ†_a U_x χ_b，两个方向
                if ((mask >> b) & 1) and not ((mask >> a) & 1):
                    nm = mask ^ (1 << a) ^ (1 << b)
                    nE = list(E)
                    nE[x] += 1 if a == x else -1    # 只动链接 x：a = x 抬、b = x 降
                    j = index.get((nm, tuple(nE)))
                    if j is None:                   # 目标态被截断：整对丢掉
                        continue
                    amp = -t
                    if boundary == "jw" and x == N - 1:
                        amp *= _jw_wrap_sign(mask, N)
                    H[i, j] += amp
    return H, states, index


def _directed_hops(N: int, states, index):
    """定向跃迁 `B_x`：把粒子从 x+1 搬到 x（即把链接 x 的电场抬 1）。H_hop = −t Σ_x (B_x + B_x†)。

    单独拿出来是为了把「E 的记账」钉死：链接上的流 j_x = −i t (B_x − B_x†) 必须满足
    离散 Maxwell 方程 `i[H, E_x] = j_x`——这正是旧写法坏掉的地方（抬错链接）。
    """
    dim = len(states)
    B = [np.zeros((dim, dim)) for _ in range(N)]
    for i, (mask, E) in enumerate(states):
        for x in range(N):
            y = (x + 1) % N
            if ((mask >> y) & 1) and not ((mask >> x) & 1):
                nm = mask ^ (1 << x) ^ (1 << y)
                nE = list(E)
                nE[x] += 1
                j = index.get((nm, tuple(nE)))
                if j is not None:
                    B[x][j, i] = 1.0
    return B


def _permutation(N: int, states, index, step: int):
    """平移在基上的置换 `perm[i] = i 的 step 格像`（不封闭就报错，不静默丢态）。"""
    perm = []
    for mask, E in states:
        nm, nE = _shift(N, mask, E, step)
        j = index.get((nm, nE))
        if j is None:
            raise ValueError(f"平移不封闭：step={step} 下 (mask={mask}, E={E}) 的像不在基里")
        perm.append(j)
    return perm


def translation(N: int, states, index, step: int = 2):
    """平移算符 `T = Σ_i |perm(i)⟩⟨i|`：(mask, E) → (rot_step mask, roll_step E) 的**纯置换**。

    不加任何相位：态依赖的相位（1 / (−1)^{n_0} / (−1)^{n_{N−1}} / (−1)^{popcount} /
    (−1)^{奇数格}）实测都把 ‖[H,T]‖ 顶到 0.64–2.0，而常数相位只是整体重定义、对
    [H,T] 与 T^{N} = 1 都无影响，所以置换本身就是唯一的答案。置换矩阵天然幺正、阶整除 N/gcd(N,step)。
    """
    dim = len(states)
    T = np.zeros((dim, dim))
    for i, j in enumerate(_permutation(N, states, index, step)):
        T[j, i] = 1.0
    return T


def momentum_sectors(N: int, states, index, step: int = 2):
    """按 T 的轨道做 Fourier 变换，返回 `(labels, U)`。

    `U` 的列是 T 的本征矢：对长度 L 的轨道，(m, L) 给
    `T|col⟩ = e^{2πi·m/L}|col⟩`；`labels[a] = (m, L)` 已约分，所以同一动量（同一 T 本征值，
    可能来自不同长度的轨道）落在同一个标签上。格点动量 `k = π·m/L`（step = 2；k 与 k+π
    是同一个 T 本征值，故 k 折叠在 [0, π)）。

    与 H 无关地建；`[H,T] = 0` ⟺ `U†HU` 只在标签相同的列之间非零——这就是「按 k 分扇区」
    的可用形式（后面测 ω(k) 就是在这些块里取最低激发）。
    """
    dim = len(states)
    perm = _permutation(N, states, index, step)
    seen = [False] * dim
    cols, labels = [], []
    for start in range(dim):
        if seen[start]:
            continue
        orbit, i = [], start
        while not seen[i]:
            seen[i] = True
            orbit.append(i)
            i = perm[i]
        L = len(orbit)
        for mm in range(L):
            col = np.zeros(dim, dtype=complex)
            for t, idx in enumerate(orbit):
                col[idx] = np.exp(-2j * np.pi * mm * t / L)
            cols.append(col / np.sqrt(L))
            gg = int(np.gcd(mm, L))
            labels.append((mm // gg, L // gg))
    return labels, np.column_stack(cols)


def _label_codes(labels) -> np.ndarray:
    """把 (m, L) 标签压成整数编号，便于一次算出「跨扇区」的掩码。"""
    table, code = {}, []
    for lab in labels:
        if lab not in table:
            table[lab] = len(table)
        code.append(table[lab])
    return np.array(code, dtype=int)


def check_basis(N: int = 8, Emax: int = 2, gs=(0.2, 0.4, 1.0), m: float = 0.0,
                t: float = 0.5, step: int = 2, sizes=(10,)) -> dict:
    """P41 的验收：平移严格对易、T 的群性质、厄米、Gauss 逐态、动量扇区分块对角。

    返回的每一项默认都必须是**残差**（判据 < 1e-5）；只作读数的数字以 `_magnitude` 结尾。
    """
    states, index = enumerate_basis(N, Emax)
    dim = len(states)
    out = {"基非空": 0.0 if dim else 1.0}
    out["维数_magnitude"] = float(dim)
    if not dim:
        return out

    gauss = 0.0
    for mask, E in states:
        r = _rho(N, mask)
        gauss = max(gauss, max(abs(E[x] - E[(x - 1) % N] - r[x]) for x in range(N)))
    out["高斯定律逐态成立"] = float(gauss)

    herm = 0.0
    Hs = {}
    for g in gs:
        H, _, _ = build_h(N, g, m, t, Emax)
        Hs[g] = H
        herm = max(herm, float(np.abs(H - H.T).max()))
        T = translation(N, states, index, step)
        out[f"[H,T]（g={g}）"] = float(np.abs(H @ T - T @ H).max())
    out["H 厄米（‖H−Hᵀ‖）"] = herm

    T = translation(N, states, index, step)
    eye = np.eye(dim)
    out["T 幺正"] = float(np.abs(T.T @ T - eye).max())
    out["T 是置换（每列恰一个 1）"] = float(np.abs(T.sum(axis=0) - 1.0).max())
    out["T^N = 1"] = float(np.abs(np.linalg.matrix_power(T, N) - eye).max())
    order = next(k for k in range(1, N + 1) if np.abs(np.linalg.matrix_power(T, k) - eye).max() == 0.0)
    out["T^{N/2} = 1"] = float(np.abs(np.linalg.matrix_power(T, N // 2) - eye).max()) if N % 2 == 0 else 0.0
    out["T 的阶_magnitude"] = float(order)

    # 矩阵元协变：H[i,j] 与 H[P i, P j] 逐条相等（旧写法在这里剩 73 条）
    perm = _permutation(N, states, index, step)
    H0 = Hs[gs[0]]
    Hp = H0[np.ix_(perm, perm)]
    nz = H0 != 0.0
    out["矩阵元错配条数（H[i,j] vs H[Pi,Pj]）"] = float(np.sum(Hp[nz] != H0[nz]))

    # 离散 Maxwell 方程 i[H, E_x] = j_x：把跳跃的 E 记账（抬哪根链接）钉死
    Bx = _directed_hops(N, states, index)
    Eflat = np.array([[E[x] for x in range(N)] for _mask, E in states], dtype=float)
    Hm = build_h(N, gs[0], m, t, Emax)[0]
    maxw = 0.0
    for x in range(N):
        jx = -1j * t * (Bx[x] - Bx[x].T)
        dh = 1j * (Hm * np.subtract.outer(Eflat[:, x], Eflat[:, x]))
        maxw = max(maxw, float(np.abs(dh - jx).max()))
    out["i[H,E_x] = j_x（离散 Maxwell）"] = maxw

    # 动量扇区：U†HU 只在同一标签的列之间非零
    labels, U = momentum_sectors(N, states, index, step)
    codes = _label_codes(labels)
    B = U.conj().T @ H0 @ U
    cross = codes[:, None] != codes[None, :]
    out["动量扇区分块对角（跨扇区最大矩阵元）"] = float(np.abs(B[cross]).max()) if cross.any() else 0.0
    phase = np.array([np.exp(2j * np.pi * p / q) for p, q in labels])
    out["T 在扇区基下对角（相位）"] = float(np.abs(U.conj().T @ T @ U - np.diag(phase)).max())
    out["动量扇区数_magnitude"] = float(len(set(labels)))
    kgrid = sorted({round(np.pi * p / q, 12) for p, q in labels})
    out["k 网格点数_magnitude"] = float(len(kgrid))

    # 对照（不参与判定）：1 格平移不保 Gauss，JW 串边界破坏平移但谱不变
    moved = 0
    for mask, E in states:
        if index.get(_shift(N, mask, E, 1)) is None:
            moved += 1
    out["1 格平移的像不在基里_magnitude"] = float(moved)
    Hjw, _, _ = build_h(N, gs[-1], m, t, Emax, boundary="jw")
    Tjw = translation(N, states, index, step)
    out["JW 串边界的 ‖[H,T]‖_magnitude"] = float(np.abs(Hjw @ Tjw - Tjw @ Hjw).max())
    w_u = np.linalg.eigvalsh(build_h(N, gs[-1], m, t, Emax)[0])
    out["JW 串边界与均匀边界同谱（max|ΔE|）"] = float(np.abs(w_u - np.linalg.eigvalsh(Hjw)).max())

    # 换一个尺寸复核：结论不能是 N = 8 的巧合（N=10 约 0.9 s，N=12 约 12 s，所以只放 N=10 进门）
    for N2 in sizes:
        H2, st2, idx2 = build_h(N2, gs[-1], m, t, Emax)
        T2 = translation(N2, st2, idx2, step)
        perm2 = _permutation(N2, st2, idx2, step)
        H2p = H2[np.ix_(perm2, perm2)]
        nz2 = H2 != 0.0
        out[f"[H,T]（N={N2}, g={gs[-1]}）"] = float(np.abs(H2 @ T2 - T2 @ H2).max())
        out[f"矩阵元错配条数（N={N2}）"] = float(np.sum(H2p[nz2] != H2[nz2]))
        out[f"T^N = 1（N={N2}）"] = float(np.abs(np.linalg.matrix_power(T2, N2) - np.eye(len(st2))).max())
        lab2, U2 = momentum_sectors(N2, st2, idx2, step)
        code2 = _label_codes(lab2)
        B2 = U2.conj().T @ H2 @ U2
        cross2 = code2[:, None] != code2[None, :]
        out[f"动量扇区分块对角（N={N2}）"] = float(np.abs(B2[cross2]).max()) if cross2.any() else 0.0
        out[f"维数（N={N2}）_magnitude"] = float(len(st2))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check_basis(), ensure_ascii=False, indent=2, default=float))
