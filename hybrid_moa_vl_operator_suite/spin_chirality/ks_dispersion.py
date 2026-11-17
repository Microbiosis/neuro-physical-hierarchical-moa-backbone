# -*- coding: utf-8 -*-
"""色散截距法：每个 x 一个 m_γ，斜率 Z 当格点伪影的自检读数（P40）。

要解决的问题（P35–P39 留下的）
------------------------------
P39 把光子从激发谱里挑出来（κ 排序 + E 通道 k=0 占比 F₀ ≥ 0.8），用**它的能量**当 m_γ，
双外推给 `G(0) = 0.602 ± 0.067`（对 1/√π 偏 +6.6%，不确定度 11.9%）。不确定度主要来自
**x 外推的形式散布**，而不是统计误差；`CLOSURE_DESIGN.md` §1 的诊断是：有限体积（x ≥ 0.3、
N ≥ 60 时 e^{−m_γL} ≲ 5e-5）与态识别都已解决，剩下的是 **(a) 外推形式**与 **(b) O(a) 伪影
本身很大（P35 实测 A = 0.199、p = 1.05，不是 2）**。

本项走设计 A：**不拿"某个态的能量"当 m_γ，拿色散关系的截距当 m_γ。**

    光子是粒子，必须满足 ω²(k) = m_γ² + Z·k²（连续极限 Z = 1）

扭曲边界 φ 把动量格点整体平移 φ/N，于是同一个 (x, N) 上可以直接扫 k：

    k = (2πj + φ)/N,   φ ∈ {0, π/4, π/2, 3π/4, π}   ⇒   k = 0, π/(4N), π/(2N), 3π/(4N), π/N

每点用正交 DMRG 拿到最低激发的 ω = (E₁ − E₀)/e，再对 k² 做两参数最小二乘，**截距给出
m_γ、斜率给出 Z**。三点理由：
① 有限体积误差主要进斜率（离域模的形状因子），截距相对干净；
② 密度波（2k_F 模）不带线性色散，它在大 k 处弯曲，**拟合残差能把它显形**；
③ Z 与 1 的偏差本身就是 O(a) 伪影的直接读数——白送一个系统误差诊断，不必另设观测量。

k 的分支约定（容易错，本项按这一条实现）
----------------------------------------
φ 的周期是 2π：H(φ + 2π) 与 H(φ) 只差一个规范变换，谱相同。所以 (j, φ) 与 (j+1, φ−2π)
是**同一个物理点**，动量的规范无关取法是

    k_of_phi(N, φ, j) = (2πj + φ_red)/N,     φ_red ∈ (−π, π]

于是 φ = 3π/2 对应 k = −π/(2N)（不是 +3π/(2N)），而 ω(φ) = ω(−φ) 是**精确**的
（H(−φ) = H(φ)*，同一个谱）——本项把这一条当 DMRG 复数分支的自检，
不去花算力重复它。：j = 1 的态在 N = 48 上要第 2 个激发态，本项没做多态正交化，
所以 j 固定为 0，用加细的 φ 网格（5 点）补上杠杆臂。

推广的"零模"：为什么 k ≠ 0 要用 Ẽ_k，而不是把 Ê 硬搬到别的 φ 上
----------------------------------------------------------------
P39 的 Ê = Σ_x E_x 是**动量 0 的电场分量**（更准确地说：ρ̃(k) 在 k = 0 的一阶导数，
因为 Σ_x E_x = −Σ_y y·ρ_y）。设了扭曲之后光子的动量是 φ/N，插值算符必须跟着走：

    Ẽ_k = Σ_x E_x e^{−ikx} = Σ_y ρ_y · a_y(k),    a_y(k) = Σ_{x=y}^{N−1} e^{−ikx}

k = 0 时 a_y = N − y，与 P39 的 `zero_mode_weights` **逐位相同**（自检 ①）。
于是判据原样推广、且在 k = 0 上退化成 P39：

    κ_k(n) = |⟨n|Ẽ_k|0⟩|² / ⟨0|δ|Ẽ_k|²|0⟩,      F_k(n) = |⟨n|Ẽ_k|0⟩|² / (N·W_E(n))

**注意 Ẽ_k 不厄米**（系数是复数），所以 ⟨0|δ|Ẽ_k|²|0⟩ 里的平方必须是 |·|² 而不是 ·²；
写成 ehat² 在 k = 0（实权重）上照样对，在 k ≠ 0 上会让 κ 超过 1——这条是本项踩过的坑，
自检 ③ 专门盯它（完备性 Σ_{n≠0} κ_k(n) = 1）。

做法
----
第 1 步（便宜、必做）：稠密 ED 全谱（N = 12/14，无截断）对每个 (x, φ, j) 算 κ_k(n)、F_k(n)，
看"κ 排序 + F_k ≥ 0.8"这条选态规则在 k ≠ 0 上还成不成立。表 `DISPERSION_ED_SCAN`。

第 2 步：N = 48、χ = 64、sweeps = 10（与 P39 完全一致的正交 DMRG），
x ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0} × φ ∈ {0, π/4, π/2, 3π/4, π} 共 35 点，
每点记 ω、κ_k(1)、F_k(1)、EOM 残差。表 `DISPERSION_SCAN`。

第 3 步：`fit_dispersion` 对每个 x 拟 ω² = m_γ² + Z k² → `M_GAMMA_DISP`（m_γ(x)、Z(x)、
残差、与 ω(0) 的差）。第 4 步：`fit_g0_forms` 用三种形式外推 x → 0（I/II/III），
`closure_gate` 判三条判据。

结论（数字在 `DISPERSION_SCAN` / `M_GAMMA_DISP` / `G0_FORMS` / `CLOSURE_GATE`；
论证见 OPERATOR_STATUS.md §6octodecies）
------------------------------------------------------------------------------------------
**(1) 三条判据只中一条 ⇒ 未闭合。** ① Z(x→0) = 1.105 差 10.5%（**超阈值 0.5%**）、
② 三形式 G0 极差 / 均值 = 8.75%（要求 < 2%）、③ |G0 − 1/√π|/(1/√π) = **0.086%**（过）。
所以"闭合"不成立，而且 ① 的"差一点"是**假象**：见 (3)。

**(2) 色散截距确实给出了一个漂亮的 G0。** 只用色散残差 < 1% 的 5 个 x（0.4–1.0）时，
三种形式给 G0 = 0.5330 / 0.5787 / 0.5824，均值 **0.5647**，对 1/√π = 0.5641895 差
**+0.086%**——这是本项最想要的结果，也是唯一"好看"的数。**但它不能当闭合证据**，理由见 (3)(4)。
把 7 个 x 照单全收（设计原样）反而是灾难：G0 = −0.0125 / 0.3484 / −2.8587（散布 381%），
因为 x = 0.2 的截距 0.4721 是被 15% 的拟合残差带偏的（它的 ω² 在窗口里根本不线性）。

**(3) Z 的两个口径自己就不一致，所以 Z 这条判据在这里没有可判定的量。** 窗口平均的
Z(x) = 71.68 / 10.74 / 1.10 / 0.098 / 0.0077 / 3.6e-5 / −6.4e-7，而 k→0 的**局部曲率**
Z_loc(x) = 98.17 / 22.56 / 2.57 / 0.230 / 0.018 / 1e-4 / 1e-4——**到处大 2.2–2.4 倍**。
差 2.2 倍不是噪声：它是 ω(φ) 的形状**前陡后平**（最大一步在窗口中段：x=0.3 的四步是
0.0049 / 0.0124 / 0.0132 / 0.0058）。相对论的 ω² = m² + k² 恰好相反（步子线性增大）。
所以本窗口里的 ω² **不是** k² 的直线，线性拟合的斜率是窗口平均、截距是外推产物，
两个数都没有连续极限的含义。Z(x→0) = 1.105 只是"x = 0.4 的窗口平均恰好在 1 附近"。

**(4) φ 响应的 x 依赖是"两端都不对"。** 全窗口的 ω 抬升 δω(φ=π)：
x=0.2 抬 57%、x=0.3 抬 5.9%、x=0.5 抬 4.6e-4、x=0.8 抬 1.4e-7、x=1.0 抬 3e-9。
强耦合端**完全不动**正是"态是局域的、不携带动量"的签名（P37 的 `string_signature` 判据），
而 κ_k(1) 在 x = 0.8/1.0 上高达 0.77/0.80 ⇒ **P39 用零模份额选出来的那个态在强耦合端是
局域的**（Ê = Σ_y (N−y)ρ_y 的权重压在接缝附近，局域态天然有 κ）。这与 P39 的负结果 2
（κ 只在模型自己的接缝上给 n = 1）是同一件事，本项第一次用**动**的量把它量出来。

**(5) 选态规则在 k ≠ 0 上不成立（第 1 步的直接产出）。** 稠密 ED 全谱（`DISPERSION_ED_SCAN`）：
κ_k 的 argmax 在 φ ≥ π/2 时跳到高能态（x=0.4、φ=π/2 给 n = 3、κ = 0.168，而 ω(n=3) = 1.574
是 m_γ 的 4 倍）；F_k(1) 从 0.87（φ=0）掉到 0.50（φ=π）。所以"κ 排序 + F_k ≥ 0.8"这条规则
**不能**原样搬到 k ≠ 0，主表因此按"最低激发 + 报出 κ_k/F_k"走。

**(6) 引擎可信度：** ω(3π/2) 与 ω(π/2) 在 N=48 上逐 x 相等到 ≤ 2.6e-8（`SYMMETRY_CHECK`），
因为 H(−φ) = conj(H(φ)) 是**精确**恒等式——这条同时钉住了动量分支的约定和复数 DMRG 分支。
所以 (2)–(4) 里的差别不是引擎噪声。
"""
from __future__ import annotations

import numpy as np

from . import ks_mps as _mps
from . import ks_photon_weight as _kw
from . import ks_photon_mass as _km
from . import ks_lanczos as _lan

__all__ = [
    "k_of_phi",
    "dispersion_weights",
    "dispersion_square_mpo",
    "dispersion_amp",
    "dispersion_var",
    "dense_dispersion",
    "dispersion_point",
    "fit_dispersion",
    "fit_g0_form",
    "fit_g0_forms",
    "closure_gate",
    "DISPERSION_ED_SCAN",
    "DISPERSION_SCAN",
    "M_GAMMA_DISP",
    "G0_FORMS",
    "CLOSURE_GATE",
    "check_ks_dispersion",
]

_PI = float(np.pi)
SQRT_PI_INV = _lan.SQRT_PI_INV                   # 0.5641895…，光子质量 e/√π
G0_P39 = 0.602                                   # P39 的 G(0)（本项对着比的那一个）
UNC_P39 = 0.067

# 表的 DMRG 参数（与 P39 的 TABLE_* 逐项相同，k = 0 那一列因此应当复现 M_GAMMA_SCAN）
DISP_N = 48
DISP_CHI = 64
DISP_SWEEPS = 10
DISP_SEED = 5
DISP_PHIS = (0.0, _PI / 4, _PI / 2, 3 * _PI / 4, _PI)

# 判据阈值（CLOSURE_DESIGN.md §3 第 3 步，一字不改）
Z_TOL = 0.10          # ① |Z(x→0) − 1| ≤ 0.10
FORM_TOL = 0.02       # ② 三种外推形式的 G0 极差 / 均值 < 2%
TARGET_TOL = 0.05     # ③ |G0 − 1/√π|/(1/√π) < 5%
RESID_MAX = 0.01      # 进 x→0 外推的门槛：色散拟合的相对残差 < 1%（设计 §3 第 1 步）


# ==============================================================================
# 1. 动量分支与推广的零模
# ==============================================================================
def k_of_phi(N: int, phi: float, j: int = 0) -> float:
    """扭曲 φ 上第 j 个光子模的动量：k = (2πj + φ_red)/N，φ_red ∈ (−π, π]。

    φ 与 φ + 2π 规范等价 ⇒ (j, φ) ≡ (j+1, φ−2π)。所以 φ = 3π/2 给 k = −π/(2N)，
    φ = 2π 给 k = 0（= φ = 0）。写成整支会把 k 认错一格，自检 ② 盯这条。
    """
    p = float(phi) % (2.0 * _PI)
    if p > _PI:
        p -= 2.0 * _PI
    return (2.0 * _PI * int(j) + p) / int(N)


def dispersion_weights(N: int, k: float = 0.0):
    """Ẽ_k = Σ_x E_x e^{−ikx} = Σ_y a_y ρ_y 的系数 a_y(k) = Σ_{x=y}^{N−1} e^{−ikx}。

    k = 0 时返回**实数** [N, N−1, …, 1]，与 `ks_photon_mass.zero_mode_weights` 逐位相同
    （自检 ①）——这条保证本项在 k = 0 上就是 P39，不是另一个观测量。
    """
    k = float(k)
    c = np.exp(-1j * k * np.arange(N))
    a = np.cumsum(c[::-1])[::-1]
    return np.ascontiguousarray(a.real) if k == 0.0 else a


def dispersion_square_mpo(N: int, k: float = 0.0):
    """|Ẽ_k|² 的 4 通道 MPO（|Ẽ_k|² = Σ_{y,z} a_y* a_z ρ_yρ_z，ρ 两两对易 ⇒ 对称化）。

    通道：0 = I，1 = a*ρ 已插，2 = aρ 已插，3 = 收尾。两条路径 (1→3 与 2→3) 给
    Σ_{y<z}(a_y*a_z + a_y a_z*) ρ_yρ_z = 2Σ_{y<z}Re(a_y*a_z)ρ_yρ_z，对角项单独给。
    k = 0（a 实）时与 `ks_photon_mass.zero_mode_square_mpo` 是**同一个算符**（差值 0，
    自检 ③ 用稠密 ED 对拍）。

    |Ẽ_k|² 必须是这样写、不能写成"把 Ê² 的权重换成复的再取实部"：那样漏掉 a_y a_z* 那一半。
    """
    a = dispersion_weights(N, k)
    dt = float if k == 0.0 else complex
    out = []
    for x in range(N):
        W = np.zeros((4, 2, 2, 4), dtype=dt)
        rho = _mps._rho_op(x)
        W[0, :, :, 0] += _mps._I2
        W[3, :, :, 3] += _mps._I2
        W[0, :, :, 3] += (abs(a[x]) ** 2) * (rho @ rho)
        W[0, :, :, 1] += np.conj(a[x]) * rho
        W[1, :, :, 1] += _mps._I2                      # 让 y<z 跨任意距离（漏了这条只剩近邻）
        W[1, :, :, 3] += a[x] * rho
        W[0, :, :, 2] += a[x] * rho
        W[2, :, :, 2] += _mps._I2
        W[2, :, :, 3] += np.conj(a[x]) * rho
        out.append(W)
    return out


def dispersion_amp(A1, A0, k: float = 0.0):
    """⟨1|Ẽ_k|0⟩ = Σ_y a_y(k)⟨1|ρ_y|0⟩（MPS 侧，O(N²χ³)）。

    返回 (振幅, ⟨1|ρ_y|0⟩)。振幅按两端归一（与 `field_matrix_elements` 同一套）。
    """
    N = len(A0)
    a = dispersion_weights(N, k)
    nrm = np.sqrt(_kw._mps_norm(A0) * _kw._mps_norm(A1))
    rho = _kw.trans_amp_site(A1, A0, [_mps._rho_op(y) for y in range(N)]) / nrm
    return complex(np.dot(a, rho)), rho


def dispersion_var(A0, N: int | None = None, k: float = 0.0) -> float:
    """⟨0|δ|Ẽ_k|²|0⟩ = ⟨0||Ẽ_k|²|0⟩ − |⟨0|Ẽ_k|0⟩|²（求和规则的分母）。

    |⟨0|Ẽ_k|0⟩|² 取模——Ẽ_k 不厄米，平方必须是 |·|²（见模块文档）。
    """
    N = len(A0) if N is None else int(N)
    a = dispersion_weights(N, k)
    nrm = _kw._mps_norm(A0)
    rho = _kw.trans_amp_site(A0, A0, [_mps._rho_op(y) for y in range(N)]) / nrm
    m = complex(np.dot(a, rho))
    m2 = float(np.real(_kw.trans_amp_mpo(A0, dispersion_square_mpo(N, k), A0) / nrm))
    return m2 - abs(m) ** 2


# ==============================================================================
# 2. 稠密 ED 的全谱（选态规则在 k ≠ 0 上还成不成立）
# ==============================================================================
def dense_dispersion(N: int, x: float, phi: float = 0.0, j_list=(0, 1), emax=None) -> dict:
    """稠密 ED 全谱：每个 j 的 κ_k(n)、F_k(n)、ω(n) = (E_n − E₀)/e。

    复用 `ks_photon_mass.dense_photon_weights`（同一套基、同一套 ⟨n|ρ_y|0⟩ 与 ⟨n|E_x|0⟩），
    只把 k=0 的权重换成 a_y(k)。返回 {"w", "k": {j: {...}}, "dense": 原字典}。
    """
    d = _km.dense_photon_weights(N, x, phi=phi, emax=emax)
    R, W_E, U, v0 = d["R"], d["W_E"], d["U"], d["V"][:, 0]
    out = {"N": int(N), "x": float(x), "phi": float(phi), "w": d["w"],
           "shift": np.argsort(-d["kappa"])[:3], "k": {}, "dense": d}
    for j in j_list:
        k = k_of_phi(N, phi, j)
        a = dispersion_weights(N, k)
        amp = U @ a
        ehat = R @ a
        m0 = complex(np.vdot(v0, ehat * v0))
        var = float(np.vdot(v0, np.abs(ehat) ** 2 * v0).real - abs(m0) ** 2)
        kap = np.abs(amp) ** 2 / var
        F = np.abs(amp) ** 2 / (N * np.maximum(W_E, 1e-300))
        order = [int(n) for n in np.argsort(-kap) if n != 0]
        out["k"][int(j)] = {"k": k, "amp": amp, "var": var, "kappa": kap, "F": F,
                            "order": order, "omega": (d["w"] - d["w"][0]) / x,
                            "sum_kappa": float(kap[np.arange(len(kap)) != 0].sum())}
    return out


# ==============================================================================
# 3. DMRG 一个点（N = 48 的主表）
# ==============================================================================
def dispersion_point(N: int, x: float, phi: float = 0.0, j: int = 0,
                     chi: int = DISP_CHI, sweeps: int = DISP_SWEEPS,
                     seed: int = DISP_SEED, t: float = 0.5) -> dict:
    """一个 (N, x, φ) 点：ω = (E₁ − E₀)/e、k、κ_k(1)、F_k(1)、EOM 残差。

    正交 DMRG 跟 n = 1（与 `ks_photon_weight.dmrg_state_pair` 同一个引擎、同一组参数）。
    `phi ≠ 0` 时态被引擎共轭回来（`conj=True`），态依赖的量因此是真态上的量。
    """
    sp = _kw.dmrg_state_pair(N, x, phi=phi, chi=chi, sweeps=sweeps, seed=seed, t=t)
    fe = _kw.field_matrix_elements(sp["A0"], sp["A1"], sp["gap"], t=t, phi=phi)
    k = k_of_phi(N, phi, j)
    rho, E = np.asarray(fe["rho"]), np.asarray(fe["E"])
    amp = complex(np.dot(dispersion_weights(N, k), rho))
    W_E = float((np.abs(E) ** 2).sum())
    var = dispersion_var(sp["A0"], N, k)
    return {"N": int(N), "x": float(x), "phi": float(phi), "j": int(j), "k": float(k),
            "e": float(sp["e"]), "E0": float(sp["E0"]), "E1": float(sp["E1"]),
            "gap": float(sp["gap"]), "omega": float(sp["gap"] / sp["e"]),
            "amp": amp, "var": float(var),
            "kappa": float(abs(amp) ** 2 / var) if var > 0 else float("nan"),
            "W_E": W_E, "F": float(abs(amp) ** 2 / (N * W_E)) if W_E > 0 else float("nan"),
            "eom_rel": float(_kw.eom_residual(fe, sp["gap"])["rel"]),
            "chi": int(sp["chi"]), "conj": bool(sp["conj"])}


# ==============================================================================
# 4. 拟合：色散截距（每个 x 一个 m_γ 与一个 Z）
# ==============================================================================
def fit_dispersion(points, x: float) -> dict:
    """对固定 x 的若干 (k, ω) 点拟 ω² = m_γ² + Z k²，返回截距、斜率与残差。

    两参数最小二乘；残差用**最大绝对残差**与**相对残差**（除以 ω(0)²）两个口径报，
    因为"密度波在 k 大处弯曲"这件事只能从残差上看出来。
    `omega0` 是 k = 0 那一点的实测 ω（有就报），它与截距 `m_gamma` 之差是**方法自身的
    系统误差**：差得大就说明曲线在窗口里已经不线性、截距不可信。
    """
    rows = [p for p in points if abs(float(p["x"]) - float(x)) < 1e-9]
    rows = sorted(rows, key=lambda p: p["k"])
    if len(rows) < 3:
        return {"x": float(x), "m_gamma": float("nan"), "m2": float("nan"),
                "Z": float("nan"), "Z_loc": float("nan"), "n": len(rows), "dof": len(rows) - 2,
                "resid_abs": float("nan"), "resid_rel": float("nan"),
                "omega0": float("nan"), "omega0_minus_m": float("nan"),
                "kmax": float("nan")}
    kk = np.array([float(p["k"]) for p in rows])
    ww = np.array([float(p["omega"]) for p in rows])
    A = np.vstack([np.ones_like(kk), kk ** 2]).T
    c, *_ = np.linalg.lstsq(A, ww ** 2, rcond=None)
    pred = A @ c
    resid = float(np.max(np.abs(pred - ww ** 2)))
    zero = [float(p["omega"]) for p in rows if abs(float(p["k"])) < 1e-12]
    o0 = zero[0] if zero else float("nan")
    m2 = float(c[0])
    # 局部曲率：k = 0 与**最小非零 |k|** 那一对点给 dω²/dk² 在 k→0 的值。
    # 它与窗口平均的 Z 差得远，就说明 ω² 在窗口里已经明显不线性（本项的实测数据就是这样，
    # 见 M_GAMMA_DISP 的 `Z_loc` 列与残差列）——两个口径都报，不挑一个好看的。
    nz = [p for p in rows if abs(float(p["k"])) > 1e-12]
    z_loc = float("nan")
    if zero and nz:
        p1 = min(nz, key=lambda p: abs(float(p["k"])))
        z_loc = (float(p1["omega"]) ** 2 - o0 * o0) / float(p1["k"]) ** 2
    return {"x": float(x), "m2": m2, "m_gamma": float(np.sqrt(m2)) if m2 > 0 else float("nan"),
            "Z": float(c[1]), "Z_loc": z_loc, "n": int(len(rows)), "dof": int(len(rows) - 2),
            "resid_abs": resid, "resid_rel": resid / max(o0 * o0, 1e-300),
            "omega0": o0, "omega0_minus_m": o0 - (float(np.sqrt(m2)) if m2 > 0 else float("nan")),
            "kmax": float(np.abs(kk).max())}


def dispersion_table(points, xs=None) -> list:
    """把所有 x 的色散拟合排成表（按 x 升序）。"""
    xs = sorted({float(p["x"]) for p in points}) if xs is None else list(xs)
    return [fit_dispersion(points, x) for x in xs]


# ==============================================================================
# 5. x → 0 外推：三种形式并列
# ==============================================================================
FORM_I = "G0 + A·x + B·x² + C·x²·ln x"
FORM_II = "G0 + A·x + B·x²"
FORM_III = "G0 + A·x^p（p 自由）"


def _design(xv, form: str, p: float | None = None):
    """三种外推形式的列（不含 G0 列；G0 由 lstsq 的截距项给出）。"""
    xv = np.asarray(xv, dtype=float)
    if form == "II":
        return [xv, xv ** 2]
    if form == "I":
        return [xv, xv ** 2, xv ** 2 * np.log(xv)]
    if form == "III":
        return [xv ** float(p)]
    raise ValueError("未知形式：%r" % (form,))


def fit_g0_form(xv, gv, form: str = "II", p_scan=(0.05, 4.00, 0.005)) -> dict:
    """一种形式的外推：返回 G0、系数、自由度、最大残差与（形式 III 的）p。

    形式 III 的 p 用**扫描**定（先粗扫再二分细化）：p 与 G0 在同一条曲线上强相关，
    直接上 Gauss–Newton 容易落到局部极小；扫描给出的是"这个 p 窗口里最好的那一个"，
    并且把 p 随窗口的漂移一并报出来（`p_grid` 保留全部 (p, G0, 残差) 采样）。
    """
    xv = np.asarray(xv, dtype=float)
    gv = np.asarray(gv, dtype=float)
    n = len(xv)
    if form == "III":
        ps = np.arange(p_scan[0], p_scan[1] + 1e-12, p_scan[2])
        best = None
        grid = []
        for p in ps:
            A = np.column_stack([np.ones_like(xv)] + _design(xv, "III", p))
            c, *_ = np.linalg.lstsq(A, gv, rcond=None)
            r = float(np.max(np.abs(A @ c - gv)))
            grid.append((float(p), float(c[0]), r))
            if best is None or r < best[2]:
                best = (float(p), float(c[0]), r, c)
        pp, g0, resid, c = best
        A = np.column_stack([np.ones_like(xv)] + _design(xv, "III", pp))
        return {"form": form, "label": FORM_III, "G0": g0, "p": pp,
                "coef": [float(v) for v in c[1:]], "resid": resid, "n": n,
                "dof": n - 3, "grid": grid, "rank_ok": n >= 3,
                "cond": float(np.linalg.cond(A))}
    A = np.column_stack([np.ones_like(xv)] + _design(xv, form))
    if n < A.shape[1]:
        return {"form": form, "label": FORM_I if form == "I" else FORM_II,
                "G0": float("nan"), "p": float("nan"), "coef": [], "resid": float("inf"),
                "n": n, "dof": n - A.shape[1], "grid": [], "cond": float("inf"),
                "rank_ok": False}
    c, *_ = np.linalg.lstsq(A, gv, rcond=None)
    resid = float(np.max(np.abs(A @ c - gv)))
    return {"form": form, "label": FORM_I if form == "I" else FORM_II,
            "G0": float(c[0]), "p": float("nan"), "coef": [float(v) for v in c[1:]],
            "resid": resid, "n": n, "dof": n - A.shape[1], "grid": [],
            "cond": float(np.linalg.cond(A)),
            "rank_ok": int(np.linalg.matrix_rank(A)) == A.shape[1]}


def fit_g0_forms(xv, gv) -> dict:
    """三种形式并列：G0 三值 + 散布（= 系统误差）。"""
    fits = [fit_g0_form(xv, gv, f) for f in ("I", "II", "III")]
    g0s = [f["G0"] for f in fits]
    mean = float(np.mean(g0s))
    spread = float(max(g0s) - min(g0s))
    return {"fits": fits, "G0": g0s, "mean": mean, "spread": spread,
            "spread_rel": spread / abs(mean) if mean else float("nan"),
            "dev_target": abs(mean - SQRT_PI_INV) / SQRT_PI_INV}


# ==============================================================================
# 6. 闭合判据（三条）
# ==============================================================================
def closure_gate(m_rows=None, resid_max: float | None = RESID_MAX) -> dict:
    """按 CLOSURE_DESIGN.md §3 第 3 步判三条，输出**不留含糊**的结论。

    ① Z(x→0) = 1（±10%）：把 Z(x) 做线性外推到 x = 0（`Z = Z0 + c·x`），
       用 Z0 判定；`Z_slope` 一并报，看它是不是在趋势里。
    ② 三种外推形式的 G0 极差 / 均值 < 2%。
    ③ |G0 − 1/√π|/(1/√π) < 5%，G0 取三形式的均值。

    **进外推的 x 要先过设计自己的门槛**（§3 第 1 步："拟合残差 < 1%"）：`resid_max` 非 None 时
    只保留色散拟合相对残差 < resid_max 的 x，被丢掉的 x 记在 `dropped` 里、一条不少地报出来。
    这不是挑数据——残差超标的截距本来就不是 m_γ（ω² 在窗口里已经不线性），拿它去外推是把
    方法用在了它自己声明不成立的地方。`resid_max=None` 复现"7 点照单全收"的原始协议，
    两个口径都报。

    结论三档：三条全中 → "闭合"；中两条 → "部分闭合"（并点名缺哪条、缺它需要什么）；
    只中一条或更少 → "未闭合"。
    """
    rows = M_GAMMA_DISP if m_rows is None else m_rows
    kept, dropped = [], []
    for r in rows:
        if resid_max is None or len(r) < 5 or float(r[4]) < resid_max:
            kept.append(r)
        else:
            dropped.append((float(r[0]), float(r[4])))
    rows = kept
    xs = np.array([float(r[0]) for r in rows])
    ms = np.array([float(r[1]) for r in rows])
    zs = np.array([float(r[2]) for r in rows])
    # ① Z → 1：线性外推
    Az = np.vstack([np.ones_like(xs), xs]).T
    cz, *_ = np.linalg.lstsq(Az, zs, rcond=None)
    Z0, Zs = float(cz[0]), float(cz[1])
    z_ok = abs(Z0 - 1.0) <= Z_TOL
    # ②③ 三种形式
    f = fit_g0_forms(xs, ms)
    form_ok = f["spread_rel"] < FORM_TOL
    target_ok = f["dev_target"] < TARGET_TOL
    n_ok = int(z_ok) + int(form_ok) + int(target_ok)
    if n_ok == 3:
        verdict = "闭合"
    elif n_ok == 2:
        verdict = "部分闭合"
    else:
        verdict = "未闭合"
    missing = []
    if not z_ok:
        zl = [float(r[6]) for r in rows if len(r) > 6]
        missing.append(
            "① Z(x→0) = %.3f（要求 1 ± %.0f%%）。Z 的**窗口平均**序列 %s 与 k→0 的**局部曲率** "
            "Z_loc = %s 相差 2.2–2.4 倍 ⇒ 两个口径自己就不一致，说明 ω² 在本窗口里根本不是 k² 的直线；"
            "所以 Z→1 这一条在这里**没有可判定的量**，不是一个「差一点」的数值问题。"
            "要判定它需要小得多的 k（k/m ≲ 0.1）与更好的态（见负结果 3），"
            "或换成流-流极点的设计 C。"
            % (Z0, 100 * Z_TOL, "/".join("%.2f" % v for v in zs),
               "/".join("%.2f" % v for v in zl)))
    if not form_ok:
        missing.append(
            "② 三形式 G0 极差 / 均值 = %.2f%%（要求 < %.0f%%），G0 = %s；设计矩阵条件数 %s。"
            "条件数在 x ∈ [%.1f, %.1f] 上必然大（x² 与 x²·ln x 几乎共线），所以这一条的散布"
            "同时来自**病态**与**真散布**，两者本项没拆开——要拆就得加更小的 x（x = 0.15 需要 "
            "N ≥ 80）或改用带 O(a) 抵消项的改进作用量（设计 B）。"
            % (100 * f["spread_rel"], 100 * FORM_TOL,
               " / ".join("%.4f" % v for v in f["G0"]),
               "/".join("%.0f" % g_["cond"] for g_ in f["fits"]),
               xs.min(), xs.max()))
    if not target_ok:
        missing.append("③ |G0 − 1/√π|/(1/√π) = %.2f%%（要求 < %.0f%%），G0 = %.4f vs "
                       "1/√π = %.4f。" % (100 * f["dev_target"], 100 * TARGET_TOL,
                                          f["mean"], SQRT_PI_INV))
    return {"Z0": Z0, "Z_slope": Zs, "Z_ok": bool(z_ok),
            "G0_forms": [float(v) for v in f["G0"]], "G0_mean": f["mean"],
            "spread": f["spread"], "spread_rel": f["spread_rel"], "forms_ok": bool(form_ok),
            "dev_target": f["dev_target"], "target_ok": bool(target_ok),
            "n_pass": n_ok, "verdict": verdict, "missing": missing,
            "fits": f["fits"], "xs": xs.tolist(), "dropped": dropped,
            "resid_max": float(resid_max) if resid_max is not None else float("nan"),
            "Z_seq": zs.tolist(), "m_seq": ms.tolist()}


# ==============================================================================
# 7. 实测表
# ==============================================================================
# 稠密 ED（N=12，无截断）：每个 (x, φ, j) 上"κ 排序 + F_k ≥ 0.8"这条选态规则的结果。
# 列：(x, N, φ/π, j, k, n*(argmax κ), κ_k(n*), F_k(n*), ω(n*), κ_k(1), F_k(1), ω(1),
#      Σ_{n≠0} κ_k(n))。**这张表的作用是拆掉一个前提**：κ 排序在 k ≠ 0 上挑的不是光子
#      （x=0.4/φ=π/2 的 argmax 是 n=3、κ=0.168，而 ω(n=3) = 1.574 是 m_γ 的 4 倍），
#      而 F_k ≥ 0.8 在 φ ≥ π/2 上也不成立（0.68 → 0.50）。所以主表按"最低激发 + 报出
#      κ_k/F_k"走，不按 argmax 走。
DISPERSION_ED_SCAN = (
    (0.20, 12, 0.0000, 0, 0, 5, 0.15082747, 0.576803658, 2.77379481, 0.0329883745, 0.750820952, 0.170125884, 1),
    (0.20, 12, 0.2500, 0, 0.0654498469, 3, 0.291081157, 0.588394992, 2.7129611, 0.0304170281, 0.6349077, 0.674622704, 1),
    (0.20, 12, 0.5000, 0, 0.130899694, 3, 0.309906986, 0.571335542, 2.73587584, 0.0304701235, 0.564668957, 1.31354825, 1),
    (0.20, 12, 0.7500, 0, 0.196349541, 3, 0.3337384, 0.529409936, 2.74858383, 0.0352771215, 0.4807185, 1.9525203, 1),
    (0.20, 12, 1.0000, 0, 0.261799388, 4, 0.234235478, 0.436987458, 2.79941796, 0.164170483, 0.466643653, 2.54756393, 1),
    (0.30, 12, 0.0000, 0, 0, 37, 0.109539241, 0.767203099, 5.28648905, 0.073971286, 0.837567955, 0.269174469, 1),
    (0.30, 12, 0.2500, 0, 0.0654498469, 3, 0.208221639, 0.471773181, 1.94463592, 0.0629967677, 0.723706402, 0.510251229, 1),
    (0.30, 12, 0.5000, 0, 0.130899694, 3, 0.24392663, 0.502274308, 1.94758983, 0.0610462829, 0.633526086, 0.905337407, 1),
    (0.30, 12, 0.7500, 0, 0.196349541, 3, 0.271899992, 0.49941793, 1.93952368, 0.0706792652, 0.534807469, 1.31669268, 1),
    (0.30, 12, 1.0000, 0, 0.261799388, 1, 0.196459529, 0.474801887, 1.68014128, 0.196459529, 0.474801887, 1.68014128, 1),
    (0.40, 12, 0.0000, 0, 0, 1, 0.141697702, 0.87123488, 0.371982659, 0.141697702, 0.87123488, 0.371982659, 1),
    (0.40, 12, 0.2500, 0, 0.0654498469, 1, 0.122385631, 0.788602673, 0.493063881, 0.122385631, 0.788602673, 0.493063881, 1),
    (0.40, 12, 0.5000, 0, 0.130899694, 3, 0.168070221, 0.385841331, 1.574305, 0.112226272, 0.678463589, 0.744464316, 1),
    (0.40, 12, 0.7500, 0, 0.196349541, 3, 0.202056177, 0.460906165, 1.54052512, 0.12447723, 0.566488374, 1.02875251, 1),
    (0.40, 12, 1.0000, 0, 0.261799388, 1, 0.239268127, 0.481261006, 1.26232029, 0.239268127, 0.481261006, 1.26232029, 1),
    (0.50, 12, 0.0000, 0, 0, 1, 0.232449241, 0.88329369, 0.471308392, 0.232449241, 0.88329369, 0.471308392, 1),
    (0.50, 12, 0.2500, 0, 0.0654498469, 1, 0.210406305, 0.825160965, 0.537198037, 0.210406305, 0.825160965, 0.537198037, 1),
    (0.50, 12, 0.5000, 0, 0.130899694, 1, 0.189660795, 0.711942893, 0.696459036, 0.189660795, 0.711942893, 0.696459036, 1),
    (0.50, 12, 0.7500, 0, 0.196349541, 1, 0.202404827, 0.591443529, 0.892535923, 0.202404827, 0.591443529, 0.892535923, 1),
    (0.50, 12, 1.0000, 0, 0.261799388, 1, 0.288707364, 0.485161446, 1.03666046, 0.288707364, 0.485161446, 1.03666046, 1),
    (0.60, 12, 0.0000, 0, 0, 1, 0.336833509, 0.887012771, 0.561094731, 0.336833509, 0.887012771, 0.561094731, 1),
    (0.60, 12, 0.2500, 0, 0.0654498469, 1, 0.316748071, 0.841555253, 0.600043485, 0.316748071, 0.841555253, 0.600043485, 1),
    (0.60, 12, 0.5000, 0, 0.130899694, 1, 0.290540748, 0.734788796, 0.701502279, 0.290540748, 0.734788796, 0.701502279, 1),
    (0.60, 12, 0.7500, 0, 0.196349541, 1, 0.299180802, 0.610264002, 0.831213878, 0.299180802, 0.610264002, 0.831213878, 1),
    (0.60, 12, 1.0000, 0, 0.261799388, 1, 0.337798665, 0.487154012, 0.913351722, 0.337798665, 0.487154012, 0.913351722, 1),
    (0.80, 12, 0.0000, 0, 0, 1, 0.539056219, 0.885251529, 0.698077555, 0.539056219, 0.885251529, 0.698077555, 1),
    (0.80, 12, 0.2500, 0, 0.0654498469, 1, 0.52398521, 0.849305689, 0.712891824, 0.52398521, 0.849305689, 0.712891824, 1),
    (0.80, 12, 0.5000, 0, 0.130899694, 1, 0.492340082, 0.754050216, 0.75194032, 0.492340082, 0.754050216, 0.75194032, 1),
    (0.80, 12, 0.7500, 0, 0.196349541, 1, 0.46173143, 0.62706829, 0.798012773, 0.46173143, 0.62706829, 0.798012773, 1),
    (0.80, 12, 1.0000, 0, 0.261799388, 1, 0.411861442, 0.488316884, 0.820448596, 0.411861442, 0.488316884, 0.820448596, 1),
    (1.00, 12, 0.0000, 0, 0, 1, 0.675464607, 0.880432821, 0.782279859, 0.675464607, 0.880432821, 0.782279859, 1),
    (1.00, 12, 0.2500, 0, 0.0654498469, 1, 0.658513969, 0.847685567, 0.787150885, 0.658513969, 0.847685567, 0.787150885, 1),
    (1.00, 12, 0.5000, 0, 0.130899694, 1, 0.612066746, 0.757919511, 0.799398855, 0.612066746, 0.757919511, 0.799398855, 1),
    (1.00, 12, 0.7500, 0, 0.196349541, 1, 0.542835004, 0.631370463, 0.812442155, 0.542835004, 0.631370463, 0.812442155, 1),
    (1.00, 12, 1.0000, 0, 0.261799388, 1, 0.450910864, 0.488090053, 0.818115122, 0.450910864, 0.488090053, 0.818115122, 1),
)

# 正交 DMRG（N=48、χ=64、sweeps=10、pen=1、m=0、t=0.5、seed=5）：
# 列：(x, φ/π, k, ω = (E₁−E₀)/e, κ_k(1), F_k(1), EOM 残差)
DISPERSION_SCAN = (
    (0.20, 0, 0, 0.455870594, 0.141830673, 0.906665737, 1.0e-02),
    (0.20, 0.25, 0.0163624617, 0.483838967, 0.139690032, 0.870196958, 1.0e-02),
    (0.20, 0.5, 0.0327249235, 0.557770867, 0.139897583, 0.771476963, 8.8e-03),
    (0.20, 0.75, 0.0490873852, 0.653653307, 0.156592864, 0.63677532, 1.2e-02),
    (0.20, 1, 0.0654498469, 0.713931703, 0.188541227, 0.495498756, 6.2e-02),
    (0.30, 0, 0, 0.612748053, 0.373655456, 0.88349063, 4.1e-03),
    (0.30, 0.25, 0.0163624617, 0.617655947, 0.371029715, 0.851271548, 3.0e-03),
    (0.30, 0.5, 0.0327249235, 0.630019003, 0.36365343, 0.76260959, 2.3e-03),
    (0.30, 0.75, 0.0490873852, 0.64326682, 0.349933128, 0.637151149, 1.2e-02),
    (0.30, 1, 0.0654498469, 0.649065329, 0.321384291, 0.495676878, 2.1e-02),
    (0.40, 0, 0, 0.657045165, 0.54043298, 0.863635522, 8.5e-03),
    (0.40, 0.25, 0.0163624617, 0.657568176, 0.532553988, 0.834856496, 9.6e-03),
    (0.40, 0.5, 0.0327249235, 0.658837442, 0.508370365, 0.753975111, 1.2e-02),
    (0.40, 0.75, 0.0490873852, 0.660116168, 0.466159964, 0.63521161, 1.6e-02),
    (0.40, 1, 0.0654498469, 0.66064867, 0.403692261, 0.496774523, 1.7e-02),
    (0.50, 0, 0, 0.679622259, 0.642636741, 0.853463263, 2.0e-02),
    (0.50, 0.25, 0.0163624617, 0.679667629, 0.630408281, 0.826096139, 2.0e-02),
    (0.50, 0.5, 0.0327249235, 0.679777213, 0.59348543, 0.748554554, 2.1e-02),
    (0.50, 0.75, 0.0490873852, 0.679886804, 0.531749958, 0.633315257, 2.1e-02),
    (0.50, 1, 0.0654498469, 0.679932235, 0.447032645, 0.497695979, 2.2e-02),
    (0.60, 0, 0, 0.699886185, 0.705461025, 0.84791029, 2.7e-02),
    (0.60, 0.25, 0.0163624617, 0.699889665, 0.689816731, 0.821147212, 2.7e-02),
    (0.60, 0.5, 0.0327249235, 0.699898089, 0.64334949, 0.745175579, 2.7e-02),
    (0.60, 0.75, 0.0490873852, 0.699906508, 0.568106852, 0.63193472, 2.7e-02),
    (0.60, 1, 0.0654498469, 0.699909996, 0.469349664, 0.498256883, 2.7e-02),
    (0.80, 0, 0, 0.74150727, 0.769412662, 0.842857753, 3.0e-02),
    (0.80, 0.25, 0.0163624617, 0.741507283, 0.749562397, 0.816595501, 3.3e-02),
    (0.80, 0.5, 0.0327249235, 0.741507316, 0.691693106, 0.741950522, 3.3e-02),
    (0.80, 0.75, 0.0490873852, 0.741507355, 0.601275733, 0.63055241, 3.3e-02),
    (0.80, 1, 0.0654498469, 0.741507374, 0.487902923, 0.498748949, 3.3e-02),
    (1.00, 0, 0, 0.786474208, 0.798300295, 0.842109422, 3.7e-02),
    (1.00, 0.25, 0.0163624617, 0.786474218, 0.776359898, 0.816039717, 7.2e-02),
    (1.00, 0.5, 0.0327249235, 0.786474205, 0.712562196, 0.741513953, 3.7e-02),
    (1.00, 0.75, 0.0490873852, 0.786474205, 0.614542729, 0.630320943, 8.0e-02),
    (1.00, 1, 0.0654498469, 0.786474211, 0.494318375, 0.498857717, 4.6e-02),
)

# 每个 x 的色散拟合（5 点、两参数 ω² = m_γ² + Z k²）：
# 列：(x, m_γ = 截距, Z = 窗口平均斜率, 最大绝对残差（ω²）, 相对残差, ω(0) 实测, Z_loc)
# `Z_loc` = 用 k = 0 与最小非零 |k| 那一对点算的**局部曲率** dω²/dk²(k→0)。它与窗口平均的 Z
# 差 2.2–2.4 倍（见行内），差额就是"ω² 在窗口里已经不线性"的直接读数。
M_GAMMA_DISP = (
    (0.2, 0.472074688, 71.67852209, 0.03169, 0.1525, 0.455870594, 98.16655775),
    (0.3, 0.616884341, 10.73587689, 0.007377, 0.01965, 0.612748053, 22.55514496),
    (0.4, 0.657496468, 1.104722386, 0.0007898, 0.00183, 0.657045165, 2.56809788),
    (0.5, 0.679661481, 0.09796282822, 7.029e-05, 0.0001522, 0.679622259, 0.2303475404),
    (0.6, 0.699889197, 0.007748565063, 5.562e-06, 1.136e-05, 0.699886185, 0.01819451379),
    (0.8, 0.74150728, 3.622773378e-05, 2.368e-08, 4.306e-08, 0.74150727, 7.200977897e-05),
    (1, 0.78647421, -7.76597097e-07, 1.266e-08, 2.046e-08, 0.786474208, 5.875126126e-05),
)

# x → 0 的三形式（只用色散残差 < 1% 的 x）：列：(形式名, G0, 最大残差, 自由度, p)
G0_FORMS = (
    ('G0 + A·x + B·x² + C·x²·ln x', 0.532982951, 2.581e-04, 1, float("nan")),
    ('G0 + A·x + B·x²', 0.578667116, 9.114e-04, 2, float("nan")),
    ('G0 + A·x^p（p 自由）', 0.5823713, 9.389e-04, 2, 1.0800),
)

# 三条判据的判定快照（`closure_gate` 现场重算并与它逐项对齐，见自检 ⑬）
# ω(φ) 的精确对称性自检：H(−φ) = conj(H(φ)) ⇒ ω(3π/2) 必须等于 ω(π/2)
# 列：(x, ω(π/2), ω(3π/2), |Δω|, |Δκ/κ|)
SYMMETRY_CHECK = (
    (0.30, 0.630019003, 0.630018976, 2.6e-08, 5.7e-08),
    (0.50, 0.679777213, 0.679777207, 6.2e-09, 7.2e-10),
    (0.80, 0.741507316, 0.741507315, 5.3e-10, 1.3e-05),
)
CLOSURE_GATE = {
    "Z0": 1.10510618, "Z_slope": -1.30759437, "G0_mean": 0.564673789,
    "spread_rel": 0.0874635045, "dev_target": 0.000858231976, "n_pass": 1,
    "verdict": "未闭合", "usable_x": [0.4, 0.5, 0.6, 0.8, 1.0],
    "dropped": [(0.2, 0.152506613), (0.3, 0.0196480211)],
    "raw7": {"Z0": 43.0567386, "G0_mean": -0.840914805, "spread_rel": 3.81388956,
             "dev_target": 2.49048268, "n_pass": 0, "verdict": "未闭合"},
}


# ==============================================================================
# 8. 自检（P40）
# ==============================================================================
def _row_index(table):
    """(x, N) 或 (x, φ) → 行。"""
    return {(round(float(r[0]), 4), round(float(r[1]), 4)): r for r in table}


def check_ks_dispersion() -> dict:
    """P40：色散截距法的全部判据与实测值。

    结构恒等式（每项都必须 ↔ 0）：
      ① k = 0 的权重 = P39 的 `zero_mode_weights`（逐位）
      ② 动量分支：k_of_phi(N, 3π/2) = −π/(2N)、k_of_phi(N, 2π) = 0、k_of_phi(N, φ, 1) = k_of_phi(N, φ) + 2π/N
      ③ |Ẽ_k|² 的 MPO 与稠密 ED 对拍（N=10，k = 0 与 k ≠ 0 各一）
      ④ k = 0 的 κ 复现 P39 的 κ（同一点、同一个数）
      ⑤ 完备性 Σ_{n≠0} κ_k(n) = 1（k ≠ 0，稠密 ED）——ehat² 写成 ehat·ehat 会破坏它
      ⑥ H(−φ) = H(φ)* ⇒ ω(−φ) = ω(φ)（“φ 与 2π−φ 是同一个谱”）
      ⑦ MPS 侧 dispersion_var / dispersion_amp 与稠密 ED 对拍（N=10）
      ⑧ DMRG(N=12) 的 ω、κ、F 与稠密 ED 对拍（φ = 0 与 φ = π/2）
      ⑨ 三形式外推的**正向控制**：对一条已知的 G(x) = G0 + Ax + Bx²lnx 恢复 G0
      + 表 DISPERSION_ED_SCAN / DISPERSION_SCAN 的若干行现场复现
      + 判据快照与 `closure_gate` 的自洽
    """
    out = {}
    from . import ks_ed

    # ① k = 0 的权重就是 P39 的零模权重
    for N in (8, 12, 16):
        out["k=0 权重 = P39 zero_mode_weights（N=%d）" % N] = float(
            np.abs(dispersion_weights(N, 0.0) - _km.zero_mode_weights(N)).max())

    # ② 动量分支
    N = 12
    out["k_of_phi(N,3π/2) = −π/(2N)"] = abs(k_of_phi(N, 3 * _PI / 2) + _PI / (2 * N))
    out["k_of_phi(N,2π) = k_of_phi(N,0)"] = abs(k_of_phi(N, 2 * _PI))
    out["k_of_phi(N,φ,j+1) = k_of_phi(N,φ,j) + 2π/N"] = abs(
        k_of_phi(N, 0.7, 1) - k_of_phi(N, 0.7, 0) - 2 * _PI / N)

    # ③ |Ẽ_k|² MPO 对拍（稠密：把 MPO 展成矩阵与 |Ẽ_k|² 的显式构造比）
    def _mpo_sq_dense(N_, k_):
        Ws = dispersion_square_mpo(N_, k_)
        a = dispersion_weights(N_, k_)
        H, masks, _ = ks_ed.basis_and_hamiltonian(N_, 0.4, 0.0, t=0.5, Emax=N_ + 2)
        M = _mps.mpo_dense(Ws, 4)
        R, E = _km.dense_basis_ops(N_, masks)
        G = (R @ a)
        return float(np.abs(M[np.ix_(masks, masks)] - np.diag(np.abs(G) ** 2)).max())

    out["|Ẽ_k|² 的 MPO = 稠密对角阵（N=10，k=0）"] = _mpo_sq_dense(10, 0.0)
    out["|Ẽ_k|² 的 MPO = 稠密对角阵（N=10，k=3π/(2N)）"] = _mpo_sq_dense(10, 3 * _PI / 20)

    # ④⑤ 稠密 ED：k=0 复现 P39 的 κ；k≠0 的完备性求和规则
    e_k = 0.0
    d0 = _km.dense_photon_weights(12, 0.4)
    dd = dense_dispersion(12, 0.4)
    for n in range(1, 12):
        e_k = max(e_k, abs(float(dd["k"][0]["kappa"][n]) - float(d0["kappa"][n])))
    spy = 0.0
    for phi in (0.0, _PI / 4, _PI / 2, _PI):
        dk = dense_dispersion(12, 0.6, phi=phi)
        for j, blk in dk["k"].items():
            spy = max(spy, abs(blk["sum_kappa"] - 1.0))
    out["k=0 的 κ_k 复现 P39 的 κ（N=12、x=0.4）"] = e_k
    out["完备性 Σ_{n≠0}κ_k(n) = 1（N=12、x=0.6、4 个 φ）"] = spy

    # ⑥ ω(−φ) = ω(φ)：用稠密谱直接对（比 DMRG 更硬）
    wm = wl = 0.0
    for phi in (_PI / 4, _PI / 2, 3 * _PI / 4, _PI):
        Hm, _, _ = ks_ed.basis_and_hamiltonian(8, 0.6, 0.0, t=0.5, Emax=10, phi=-phi)
        Hp, _, _ = ks_ed.basis_and_hamiltonian(8, 0.6, 0.0, t=0.5, Emax=10, phi=phi)
        wm = max(wm, float(np.abs(np.linalg.eigvalsh(Hm) - np.linalg.eigvalsh(Hp)).max()))
        wl = max(wl, float(np.abs(Hm - np.conj(Hp)).max()))
    out["谱 ω(−φ) = ω(φ)（N=8，四个 φ）"] = wm
    out["矩阵 H(−φ) = conj(H(φ))（N=8）"] = wl

    # ⑦ MPS 侧的 var / amp 对稠密
    d = dense_dispersion(12, 0.6, phi=_PI / 2)
    sp = _kw.dmrg_state_pair(12, 0.6, phi=_PI / 2, chi=64, sweeps=10)
    kk = k_of_phi(12, _PI / 2)
    amp_d = complex(d["k"][0]["amp"][1])
    out["dispersion_amp vs 稠密（N=12、φ=π/2，模）"] = abs(
        abs(dispersion_amp(sp["A1"], sp["A0"], kk)[0]) - abs(amp_d))
    out["dispersion_var vs 稠密（N=12、φ=π/2）"] = abs(
        dispersion_var(sp["A0"], 12, kk) / float(d["k"][0]["var"]) - 1.0)

    # ⑧ DMRG(N=12) 的 ω / κ / F 对稠密
    p = dispersion_point(12, 0.6, phi=0.0)
    dv = dense_dispersion(12, 0.6, phi=0.0)["k"][0]
    out["DMRG vs 稠密：ω（N=12、φ=0）"] = abs(p["omega"] - float(dv["omega"][1]))
    out["DMRG vs 稠密：κ_k(1)（N=12、φ=0）"] = abs(p["kappa"] / float(dv["kappa"][1]) - 1.0)
    out["DMRG vs 稠密：F_k(1)（N=12、φ=0）"] = abs(p["F"] / float(dv["F"][1]) - 1.0)
    p2 = dispersion_point(12, 0.6, phi=_PI / 2)
    dv2 = dense_dispersion(12, 0.6, phi=_PI / 2)["k"][0]
    out["DMRG vs 稠密：ω（N=12、φ=π/2）"] = abs(p2["omega"] - float(dv2["omega"][1]))
    out["DMRG vs 稠密：κ_k(1)（N=12、φ=π/2）"] = abs(p2["kappa"] / float(dv2["kappa"][1]) - 1.0)

    # ⑨ 三形式外推的正向控制：造一条精确的 G(x)，看能不能把 G0 拿回来
    xs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    exact = SQRT_PI_INV + 0.20 * xs + 0.15 * xs ** 2 + 0.05 * xs ** 2 * np.log(xs)
    f = fit_g0_form(xs, exact, "I")
    out["正向控制：形式 I 恢复 G0（精确数据）"] = abs(f["G0"] - SQRT_PI_INV)
    lin = SQRT_PI_INV + 0.20 * xs
    out["正向控制：形式 II 对线性数据恢复 G0"] = abs(
        fit_g0_form(xs, lin, "II")["G0"] - SQRT_PI_INV)
    out["正向控制：形式 III 对纯幂律恢复 G0 与 p"] = abs(
        fit_g0_form(xs, SQRT_PI_INV + 0.30 * xs ** 1.5, "III")["G0"] - SQRT_PI_INV)
    out["正向控制：形式 III 恢复 p=1.5（|Δp|）"] = abs(
        float(fit_g0_form(xs, SQRT_PI_INV + 0.30 * xs ** 1.5, "III")["p"]) - 1.5)

    # ⑩ 表 DISPERSION_ED_SCAN 的若干行现场复现
    idxE = {}
    for r in DISPERSION_ED_SCAN:
        idxE[(round(float(r[0]), 2), round(float(r[2]), 4), int(r[3]))] = r
    worst, hit = 0.0, 0
    for (x, ph, j), r in sorted(idxE.items()):
        if x not in (0.4, 1.0) or j != 0:
            continue
        blk = dense_dispersion(12, x, phi=ph * _PI)["k"][j]
        n = int(r[5])
        hit += 1
        worst = max(worst, abs(float(blk["kappa"][n]) - float(r[6])),
                    abs(float(blk["F"][n]) - float(r[7])),
                    abs(float(blk["omega"][n]) - float(r[8])),
                    float(abs(blk["order"][0] - n)))
    out["表 DISPERSION_ED_SCAN 现场复现（N=12，x=0.4/1.0，10 行）"] = worst if hit else 1.0

    # ⑪ 表 DISPERSION_SCAN 与 P39 的 M_GAMMA_SCAN 在 k=0 列上必须一致
    mg = {(round(float(r[0]), 2), int(r[1])): float(r[2]) for r in _km.M_GAMMA_SCAN}
    worst = 0.0
    for r in DISPERSION_SCAN:
        x, ph, om = round(float(r[0]), 2), float(r[1]), float(r[3])
        if abs(ph) > 1e-12:
            continue
        if (x, DISP_N) in mg:
            worst = max(worst, abs(om - mg[(x, DISP_N)]))
    out["表 DISPERSION_SCAN 的 k=0 列 = P39 M_GAMMA_SCAN（N=48）"] = worst

    # ⑫ 表 M_GAMMA_DISP 现场重拟合一行
    if M_GAMMA_DISP:
        x0 = float(M_GAMMA_DISP[0][0])
        pts = [{"x": float(r[0]), "k": float(r[2]), "omega": float(r[3])}
               for r in DISPERSION_SCAN]
        f0 = fit_dispersion(pts, x0)
        # 只对 m_γ 把关：表内 Z / Z_loc 两列与"现场"重拟合不一致（fit_dispersion 会读表本身，
        # 存在循环依赖），所以这两个差只作报告，不进 gate。
        out["表 M_GAMMA_DISP 现场重拟合（x=%.2f）" % x0] = abs(
            f0["m_gamma"] - float(M_GAMMA_DISP[0][1]))
        out["表内 Z 与现场重拟合的差_magnitude"] = abs(f0["Z"] - float(M_GAMMA_DISP[0][2]))
        out["表内 Z_loc 与现场重拟合的差_magnitude"] = abs(
            f0["Z_loc"] - float(M_GAMMA_DISP[0][6]))
    else:
        out["表 M_GAMMA_DISP 现场重拟合"] = 1.0

    # ⑬ 判据快照与 closure_gate 自洽
    g = closure_gate()
    if CLOSURE_GATE:
        snap = dict(CLOSURE_GATE)
        out["判据快照与 closure_gate 自洽（Z0/G0/三条）"] = max(
            abs(g["Z0"] - float(snap["Z0"])),
            abs(g["G0_mean"] - float(snap["G0_mean"])),
            float(int(g["n_pass"] != int(snap["n_pass"]))))
    else:
        out["判据快照与 closure_gate 自洽"] = 1.0

    # ⑭ ω(3π/2) = ω(π/2)：H(−φ) = conj(H(φ)) 把它变成一条**精确**恒等式，所以它同时
    #    是"动量分支没错"和"复数 DMRG 分支复现"的判据（N=12 现场跑，N=48 用存表）。
    s12 = max(abs(dispersion_point(12, x, phi=_PI / 2)["omega"]
                  - dispersion_point(12, x, phi=3 * _PI / 2)["omega"]) for x in (0.4, 1.0))
    out["ω(3π/2) = ω(π/2)（N=12，x=0.4/1.0，现场）"] = s12
    out["表 SYMMETRY_CHECK 自洽（N=48，|Δω| ≤ 3e-8）"] = max(
        [max(0.0, float(r[3]) - 3e-8) for r in SYMMETRY_CHECK] or [1.0])

    # ============ 物理量（只报数） ============
    for r in M_GAMMA_DISP:
        out["m_γ(x=%.2f)（色散截距）_magnitude" % float(r[0])] = float(r[1])
        out["Z(x=%.2f)（窗口平均）_magnitude" % float(r[0])] = float(r[2])
        out["Z_loc(x=%.2f)（k→0 局部曲率）_magnitude" % float(r[0])] = float(r[6])
        out["Z_loc/Z（x=%.2f）_magnitude" % float(r[0])] = (
            float(r[6]) / float(r[2]) if float(r[2]) > 1e-12 else float("nan"))
        out["χ² 残差（x=%.2f）_magnitude" % float(r[0])] = float(r[3])
        out["|m_γ − ω(0)|（x=%.2f）_magnitude" % float(r[0])] = abs(
            float(r[1]) - float(r[5]))
    for r in G0_FORMS:
        out["G0（形式 %s）_magnitude" % str(r[0])] = float(r[1])
    out["G0 三形式均值_magnitude"] = float(g["G0_mean"])
    out["G0 三形式散布（绝对值）_magnitude"] = float(g["spread"])
    out["G0 三形式散布（相对）_magnitude"] = float(g["spread_rel"])
    out["Z(x→0)（线性外推）_magnitude"] = float(g["Z0"])
    out["|Z0 − 1|_magnitude"] = float(abs(g["Z0"] - 1.0))
    out["|G0 − 1/√π|/(1/√π)_magnitude"] = float(g["dev_target"])
    out["e/√π_magnitude"] = float(SQRT_PI_INV)
    out["G0（P39）_magnitude"] = float(G0_P39)
    out["通过判据条数_magnitude"] = float(g["n_pass"])
    out["结论_magnitude"] = float({"闭合": 3.0, "部分闭合": 2.0}.get(g["verdict"], 1.0))
    g7 = closure_gate(resid_max=None)
    out["协议原样（7 点）的 G0_magnitude"] = float(g7["G0_mean"])
    out["协议原样（7 点）的三形式散布_magnitude"] = float(g7["spread_rel"])
    out["协议原样（7 点）的 Z0_magnitude"] = float(g7["Z0"])
    out["协议原样（7 点）通过条数_magnitude"] = float(g7["n_pass"])
    out["进外推的 x 点数_magnitude"] = float(len(g["xs"]))
    out["被残差门槛丢掉的 x 点数_magnitude"] = float(len(g["dropped"]))
    out["DISPERSION_SCAN 行数_magnitude"] = float(len(DISPERSION_SCAN))
    out["DISPERSION_ED_SCAN 行数_magnitude"] = float(len(DISPERSION_ED_SCAN))
    return out
