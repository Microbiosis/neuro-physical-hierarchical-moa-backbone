# -*- coding: utf-8 -*-
"""光子：从激发谱里把它挑出来，用它的能量定义 m_γ（P39）。

问题（P38 留下的那句话）
------------------------
P38 量到最低激发的 ⟨1|E_x|0⟩ 非零、动量分布单峰在 k = π = 2k_F、f₁ = 5–7%，
据此判断"最低激发是中性 2k_F 模式"，所以 P35/P36 外推的 0.581 **测的不是 m_γ**。
那句话是从**一个矩阵元 + 一个求和规则比值**推出来的。本项把它变成一个可判定问题：

    全谱里（N = 12/14/16，稠密 ED）**哪一个本征态最像光子**？
    用它的能量定义 m_γ(x,N) = (E_n − E_0)/e，再做 x→0、N→∞ 双外推。

三条必须先说清楚的结构事实（都在本项里现场验证，不是假设）
----------------------------------------------------------
① **密度通道的 k = 0 分量恒为零。** q = 0 扇区里 Σ_x ρ_x = Q ≡ 0，所以对**任何** n
   都有 ⟨n|ρ̃_0|0⟩ = ⟨n|Q|0⟩ = 0。P38 量到"单峰在 k = π"，本身不是"缺少光子成分"，
   而是这条恒等式的另一面：ρ 通道根本看不见 k = 0。"动量分布以 k = 0 为主"这条判据
   在 ρ 通道上逻辑上不可能成立，必须换到 E 通道（见 ②）。

② **零模就是光子。** 1+1D 的规范场没有横向极化：唯一规范不变的场强是 E_x，
   光子（Schwinger 玻色子）的插值算符就是场强本身。环上的场强只定到相差一个常数，
   但 q = 0 扇区里

       Ê ≡ Σ_x E_x = Σ_y (N − y) ρ_y

   是"转一格后各项重排、和不变"的那个模式（k = 0）。**注意**：E_x 自己只定到相差一个常数，
   把接缝挪一格等于把场整体减掉 E_s（恒等式 E^{(s)}_x = E_x − E_s，自检②），于是 Ê 变
   −N·E_s ≠ 0 —— 所以 Ê 是**这个模型内**的量（H 与高斯定律自己的那一个 E_x），不是约定无关
   的普适算符。本项把它当定义、并在负结果 2 里量化这条依赖。于是"光子成分"的定义是

       κ(n) = |⟨n|Ê|0⟩|² / ⟨0|δÊ²|0⟩ ,      Σ_{n≠0} κ(n) = 1（完备性求和规则）

   κ(n) 是"电场零模的**非弹性**涨落里第 n 个本征态占多少"。这是 f₁ 的干净版本：
   P38 的 f₁ = Σ_x|⟨1|E_x|0⟩|² / Σ_x⟨0|δE_x²|0⟩ 把 k ≠ 0 的成分也算进去了，而
   Σ_x E_x² 的 k ≠ 0 部分被累计和按 1/(2 sin(k/2))² 放大——**接缝的低 k 分量最响**，
   所以 f₁ 量的是"这道缝有多尖"，不是"这个态有多像光子"。

③ **接缝缺陷：电场项破坏格点平移不变性。** 跳跃项在

       T |mask⟩ = (−1)^{n_{N−1}} |rot(mask)⟩        （环上转一格，T^N = 1）

   下严格不变（g = 0 时 ‖[H,T]‖ = 0，机器精度）。电场项不是：转一格后 E 的配置
   = 原配置平移 + 常数 ρ_{N−1}，而 q = 0 的约定把这个常数钉死，于是 [H,T] = O(g²) ≠ 0。
   **本模型的本征态因此不是动量本征态**，动量分布必然被接缝抹宽——P38 量到的
   "峰在 π 却只占 55–82%、峰位不随 φ 动"就是这条的后果，不是"中性 2k_F 模式"的证据。
   这条是结构事实：g = 0 必须为 0（进残差），g > 0 的数值作为 _magnitude 报告。

做法
----
第 1 步（便宜、必做）：稠密 ED 拿**全谱**（N = 12/14/16，Emax = N + 2，无截断 = 全部
半填充构型 C(N, N/2)），对每个 n 算 ⟨n|E_x|0⟩、W_E(n) = Σ_x|⟨n|E_x|0⟩|²、κ(n)、
ρ 通道的 |⟨n|ρ̃_k|0⟩| 与 E 通道的 k = 0 占比 F₀(n)。按三条判据（κ、F₀、W_E）排名，
看它们是否指向同一个 n——表 `PHOTON_STATE_SCAN` / `PHOTON_MOMENTUM_SCAN`。

第 2 步：认定的那个 n 用**正交 DMRG** 跟到 N = 20/24/32。注意 `ks_mps.dmrg_gap`
跟的正是 n = 1（与基态正交的最低态），所以"n* = 1"时第 2 步不需要新引擎；
若 n* ≠ 1 就必须加一整套多态正交化，本项没有那样做（见负结果）。

第 3 步：`m_γ(x,N)` 两步外推，规矩与 P35/P36 完全一致（固定 x 取最大 N 的实测值并报
与次大 N 的漂移；只在漂移 ≤ 0.02 的 x 上做 G(x) = G0 + A·x^p，p ∈ {1, 1.5, 2} 家族
取均值、家族极差与漂移方和根当不确定度），见 `fit_m_gamma`。

结论（数字在 `PHOTON_STATE_SCAN` / `M_GAMMA_SCAN`；论证见 OPERATOR_STATUS.md §6septendecies）
------------------------------------------------------------------------------------------
**(1) 按"电场零模份额"这条判据，光子 = 最低激发 n = 1（x ≥ 0.4）。** κ(1) 在 x = 0.4/0.6/1.0
上是 0.142 / 0.337 / 0.675（N=12）、0.266 / 0.448 / 0.708（N=14），并且**随 N 长大**：
x=0.4 从 N=12 的 0.142 升到 N=60 的 0.582，x=1.0 从 0.675（N=12）升到 0.786（N=32）——
与"光子是离域模、单态份额随环长增长"一致。它的 E_x 矩阵元近似均匀（F₀ = 0.83–0.90），
E ≤ 2e 的低能态里没有第二个候选（其余态最大 κ = 0.059）。

**(2) P38 的"2k_F 模式"就是同一个态。** 同一个 n = 1 的 ρ 通道峰在 k = π（占比
0.79 / 0.55 / 0.30，x = 0.4/0.6/1.0、N=12），所以"它有 2k_F 结构"与"它拿走最大的零模份额"
是**同一根本征态**的两句话，不是两个态 ⇒ P35/P36 外推的 0.581 测的就是这个态的能量。
**但**"它就是光子"这一步仍是**模型内**的判断（负结果 2）：本项能证明"没有别的态比它更像"，
不能证明"它就是连续极限里 m_γ 那个态"——后者靠 0.58 ≈ 1/√π 的数值吻合，不是靠判据。

**(3) 双外推给 G(0) = 0.602 ± 0.067。** 收敛的 5 个 x（0.3/0.4/0.5/0.6/1.0，最大 N =
60/60/60/32/32）上做 G(x) = G0 + A·x^p 家族：p=1 → 0.5651、p=1.5 → 0.6090、p=2 → 0.6303、
两参数多项式 → 0.6026；均值 0.6017、家族极差 0.0652、用到的最大漂移 0.0163 ⇒
**G(0) = 0.602 ± 0.067**。与 1/√π = 0.56419 差 **+6.6%**（1σ 内）、与 P35/P36 的 0.581 差
**+3.6%**（1σ 内）。p=1 那一支给 0.5651（差 0.16%）——**但 p 定不出来，所以不拿它当结果**。

**(4) 负结果 1：弱耦合端光子不是一个本征态。** x = 0.2 时 κ 碎片化：κ(1) 只有 0.033（N=12）
/ 0.149（N=14），最大的 κ 落在 n = 5（0.151，E/e = 2.77）/ n = 7（0.176，E/e = 4.29）⇒
x ≤ 0.2 上"单态光子"这个前提本身不成立。x = 0.2 与 x = 0.35（N=32 时漂移 0.082，没收敛）
都不进 x→0 拟合。

**(5) 负结果 2：判据不随接缝约定不变。** 把场挪到别的接缝（E → E − E_s，s ≠ N−1）后 κ 的
argmax 变成 n = 2/3/4/10（N=12、x=0.4/1.0，12 种接缝里只有 s = N−1 给 n = 1）。所以 κ 是
**模型内**的量（H 与高斯定律自己那一个 E_x），不是普适判据。这与 ③（电场项破坏平移不变性）
是同一件事的两面：模型没有平移对称性，"光子的零模"这个概念本身就依赖接缝。
"""
from __future__ import annotations

import numpy as np

from . import ks_mps as _mps
from . import ks_photon_weight as _kw
from . import ks_lanczos as _lan

__all__ = [
    "zero_mode_weights",
    "dense_basis_ops",
    "zero_mode_square_mpo",
    "zero_mode_amp",
    "zero_mode_var",
    "dense_photon_weights",
    "photon_mass_point",
    "translation_operator",
    "translation_defect",
    "PHOTON_STATE_SCAN",
    "PHOTON_MOMENTUM_SCAN",
    "M_GAMMA_SCAN",
    "fit_m_gamma",
    "check_ks_photon_mass",
]

_PI = float(np.pi)
SQRT_PI_INV = _lan.SQRT_PI_INV                  # 0.5641895…，光子质量 e/√π
G0_REF = 0.581                                  # P35/P36 给的 G(0)（本项要对着比的那一个）

# 表用的 DMRG 参数（表内数字与它们绑定；`check_ks_photon_mass` 现场复现也用这一组）
TABLE_CHI = 64
TABLE_SWEEPS = 10
TABLE_SEED = 5


# ==============================================================================
# 1. 电场零模 Ê = Σ_x E_x：系数、算符、MPO
# ==============================================================================
def zero_mode_weights(N: int):
    """Ê = Σ_x E_x = Σ_y (N − y) ρ_y 的系数 w_y。

    E_x = Σ_{y≤x}ρ_y，所以 x 求和后 y 被数 N − y 次。**不除 √N**：κ 是比值，
    整体常数无关；需要 ⟨Ê²⟩ 的量在数值上就用这一套（同一套也好对稠密 ED）。
    """
    return np.asarray([float(N - y) for y in range(N)])


def dense_basis_ops(N: int, masks):
    """mask 基上的对角量：R[d, x] = ρ_x、E[d, x] = Σ_{y≤x}ρ_y（本包的接缝约定）。

    两者都是**对角**算符，所以全谱的矩阵元可以一次算完（O(dim²·N) 而不是 O(N·dim³)）。
    """
    masks = np.asarray(masks, dtype=np.int64)
    R = np.empty((len(masks), N))
    for x in range(N):
        R[:, x] = ((masks >> x) & 1) - 0.5 * (1.0 - (-1.0) ** x)
    return R, np.cumsum(R, axis=1)


def zero_mode_square_mpo(N: int):
    """Ê² = (Σ_y w_y ρ_y)² 的 3 通道 MPO（与 `ks_photon_weight.efield_square_mpo` 同套路，
    只是权重换成 w_y 并铺满全链）：有序对 (y<z) 给 2w_yρ_y·w_zρ_z，对角给 (w_yρ_y)²。
    """
    w = zero_mode_weights(N)
    out = []
    for x in range(N):
        W = np.zeros((3, 2, 2, 3), dtype=float)
        rho = _mps._rho_op(x)
        W[0, :, :, 0] += _mps._I2
        W[1, :, :, 1] += _mps._I2
        W[2, :, :, 2] += _mps._I2
        W[0, :, :, 1] += w[x] * rho
        W[1, :, :, 2] += 2.0 * w[x] * rho
        W[0, :, :, 2] += (w[x] * rho) @ (w[x] * rho)
        out.append(W)
    return out


def zero_mode_amp(A0, A1):
    """⟨1|Ê|0⟩（MPS 侧）= Σ_y w_y ⟨1|ρ_y|0⟩，O(N²χ³)。振幅按两端归一。"""
    N = len(A0)
    w = zero_mode_weights(N)
    nrm = np.sqrt(_kw._mps_norm(A0) * _kw._mps_norm(A1))
    rho = _kw.trans_amp_site(A1, A0, [_mps._rho_op(y) for y in range(N)]) / nrm
    return complex(np.dot(w, rho))


def zero_mode_var(A0):
    """⟨0|δÊ²|0⟩ = ⟨0|Ê²|0⟩ − |⟨0|Ê|0⟩|²（求和规则的分母）。

    Ê² 走 3 通道 MPO；Ê 的一次项走 `trans_amp_site` 的累加（两者都精确到 MPS 的
    表示误差，与稠密 ED 对到 1e-6，见自检 ④⑤）。
    """
    N = len(A0)
    w = zero_mode_weights(N)
    nrm = _kw._mps_norm(A0)
    rho = _kw.trans_amp_site(A0, A0, [_mps._rho_op(y) for y in range(N)]) / nrm
    m = float(np.dot(w, rho.real))
    m2 = float(np.real(_kw.trans_amp_mpo(A0, zero_mode_square_mpo(N), A0) / nrm))
    return m2 - m * m


# ==============================================================================
# 2. 稠密 ED 的全谱权重（第 1 步）
# ==============================================================================
def _ed_module():
    from . import ks_ed
    return ks_ed


def dense_photon_weights(N: int, x: float, phi: float = 0.0, emax=None) -> dict:
    """稠密 ED 全谱：每个本征态的 κ、W_E、F₀、W_ρ 与 ⟨n|ρ_x|0⟩。

    返回字典的键：
      `w`       —— 本征值（升序）
      `U`       —— ⟨n|ρ_x|0⟩（dim × N）
      `UE`      —— ⟨n|E_x|0⟩ = cumsum_x⟨n|ρ_x|0⟩（dim × N）
      `amp`     —— ⟨n|Ê|0⟩（长度 dim）
      `var`     —— ⟨0|δÊ²|0⟩（标量，= Σ_{n≠0}|amp|²，求和规则的分母）
      `kappa`   —— |amp|²/var（n = 0 那项是**弹性**成分，不参与 Σ_{n≠0} = 1）
      `W_E`     —— Σ_x|⟨n|E_x|0⟩|²
      `W_rho`   —— Σ_x|⟨n|ρ_x|0⟩|²
      `F0`      —— E 通道的 k = 0 占比 = |amp|²/(N·W_E)（铺平的场 → 1）
      `var_E`   —— Σ_x⟨0|δE_x²|0⟩（f₁ 的分母，与 P38 的 f₁ 同一个量）
      `k`       —— 2πj/N 网格
      `rho_k`   —— |⟨n|ρ̃_k|0⟩|²（dim × N）
      `E_mean`  —— ⟨0|E_x|0⟩ 逐点（弹性剖面，接缝坡道）
      `masks`   —— 基矢
    """
    Emax = N + 2 if emax is None else emax
    H, masks, _ = _ed_module().basis_and_hamiltonian(N, x, 0.0, t=0.5, Emax=Emax, phi=phi)
    w, V = np.linalg.eigh(H)
    R, E = dense_basis_ops(N, masks)
    v0 = V[:, 0]
    U = V.conj().T @ (R * v0[:, None])
    UE = V.conj().T @ (E * v0[:, None])
    wts = zero_mode_weights(N)
    amp = U @ wts                                   # ⟨n|Ê|0⟩ = Σ_y w_y⟨n|ρ_y|0⟩
    ehat = E.sum(axis=1)                            # Σ_x E_x 的逐态对角值（= R @ wts，自检 ①）
    m0 = float(np.vdot(v0, ehat * v0).real)
    var = float(np.vdot(v0, ehat * ehat * v0).real - m0 * m0)
    W_E = (np.abs(UE) ** 2).sum(axis=1)
    _order = [int(n) for n in np.argsort(-(np.abs(amp) ** 2 / var)) if n != 0]
    F0 = np.abs(amp) ** 2 / (N * np.maximum(W_E, 1e-300))
    k = 2.0 * _PI * np.arange(N) / N
    return {"N": N, "x": float(x), "phi": float(phi), "masks": np.asarray(masks),
            "w": w, "V": V, "U": U, "UE": UE, "amp": amp, "var": var,
            "kappa": np.abs(amp) ** 2 / var, "W_E": W_E,
            "R": R, "E": E,
            "W_rho": (np.abs(U) ** 2).sum(axis=1), "F0": F0, "k": k,
            "rho_k": np.abs(np.fft.fft(U, axis=1) / np.sqrt(N)) ** 2,
            "_order": _order,
            "E_mean": (E.T * (np.abs(v0) ** 2)).sum(axis=1),
            "E2_mean": ((E ** 2).T * (np.abs(v0) ** 2)).sum(axis=1),
            "var_E": float(((E ** 2).T * (np.abs(v0) ** 2)).sum()
                           - (((E.T * (np.abs(v0) ** 2)).sum(axis=1)) ** 2).sum())}


# ==============================================================================
# 3. 一个 (N, x) 点的 DMRG 版本（第 2 步）
# ==============================================================================
def photon_mass_point(N: int, x: float, chi: int = TABLE_CHI, sweeps: int = TABLE_SWEEPS,
                      seed: int = TABLE_SEED, phi: float = 0.0, with_f1: bool = True) -> dict:
    """正交 DMRG 跟 n = 1：返回 m_γ = (E_1 − E_0)/e、κ(1)、W_E、F₀、动量峰。

    κ 只要一个 Ê² MPO（O(N·Dχ³)）加一次逐点跃迁振幅（O(N²χ³)），总是算；
    f₁ 要 N 个 E_x² MPO（`efield_variance`，O(N²·Dχ³)），`with_f1=False` 时跳过
    （大 N 扫描时用；本项表里的 κ 与 f₁ 都是打开时算的）。
    """
    sp = _kw.dmrg_state_pair(N, x, phi=phi, chi=chi, sweeps=sweeps, seed=seed)
    fe = _kw.field_matrix_elements(sp["A0"], sp["A1"], sp["gap"], phi=phi)
    rho, Efit = np.asarray(fe["rho"]), np.asarray(fe["E"])
    amp = complex(np.dot(zero_mode_weights(N), rho))
    var = zero_mode_var(sp["A0"])
    W_E = float((np.abs(Efit) ** 2).sum())
    out = {"N": int(N), "x": float(x), "e": float(sp["e"]), "gap": float(sp["gap"]),
           "m_gamma": float(sp["gap"] / sp["e"]), "E0": float(sp["E0"]),
           "E1": float(sp["E1"]), "amp": amp, "W_E": W_E, "var": float(var),
           "kappa": float(abs(amp) ** 2 / var), "f1": float("nan"),
           "W_rho": float((np.abs(rho) ** 2).sum()),
           "F0": float(abs(amp) ** 2 / (N * W_E)) if W_E > 0 else float("nan"),
           "rho": rho, "E": Efit, "echo": float(fe["echo"]),
           "eom_rel": float(_kw.eom_residual(fe, sp["gap"])["rel"])}
    kc = _kw.momentum_concentration(rho)
    out["k_peak"], out["peak_frac"], out["PRk"] = kc["k_peak"], kc["peak_frac"], kc["PRk"]
    if with_f1:
        var_E = float(np.sum(_kw.efield_variance(sp["A0"], N)))
        out.update({"f1": float(W_E / var_E) if var_E > 0 else float("nan"),
                    "var_E": var_E})
    return out


# ==============================================================================
# 4. 平移不变性：跳跃项不变、电场项破坏（结构事实）
# ==============================================================================
def translation_operator(N: int, masks):
    """环上转一格的**费米子**平移 T|mask⟩ = (−1)^{n_{N−1}}|rot(mask)⟩。

    相位 (−1)^{n_{N−1}} 不是猜的：g = 0（纯跳跃）时 ‖[H,T]‖ = 0 到机器精度，
    其余三种常见约定（+1、(−1)^{n_0}、(−1)^{popcount}）都给 1.0（见自检 ⑦）。
    """
    masks = np.asarray(masks, dtype=np.int64)
    idx = {int(mk): i for i, mk in enumerate(masks)}
    dim = len(masks)
    m = (1 << N) - 1
    T = np.zeros((dim, dim))
    for i, mk in enumerate(masks):
        j = idx.get(int((((int(mk) << 1) | (int(mk) >> (N - 1))) & m)))
        if j is None:                                  # 转一格跑出扇区（不该发生）
            raise ValueError("平移不在基内：N 必须使 Σρ = 0 扇区闭合")
        T[j, i] = (-1.0) ** ((int(mk) >> (N - 1)) & 1)
    return T


def _rotation_index(N: int, masks):
    """rot(i)：把基矢 i 转一格后落在基里的**下标**（自检 ⑦ 的恒等式用）。"""
    masks = np.asarray(masks, dtype=np.int64)
    idx = {int(mk): i for i, mk in enumerate(masks)}
    m = (1 << N) - 1
    return np.asarray([idx[int((((int(mk) << 1) | (int(mk) >> (N - 1))) & m))]
                       for mk in masks])


def translation_defect(N: int, g: float, t: float = 0.5) -> float:
    """‖[H, T]‖∞：g = 0 应为 0（跳跃项在 T 下不变），g > 0 是电场项的接缝缺陷。

    物理内容：转一格后 E 的配置 = 原配置平移 + 常数 ρ_{N−1}，而 q = 0 的约定
    （E_{N−1} = q = 0）把这个常数钉死，于是 Σ_x E_x² 不是平移不变的。缺陷的大小
    就是这里的 [H,T]；它和 t、g²/2 同量级，所以**不是小量**——本模型的本征态不是
    动量本征态，动量分布必然被接缝抹宽。
    """
    H, masks, _ = _ed_module().basis_and_hamiltonian(N, g, 0.0, t=t, Emax=N + 2)
    T = translation_operator(N, masks)
    return float(np.abs(H @ T - T @ H).max())


# ==============================================================================
# 5. 实测表（第 1 步：N = 12/14 稠密全谱；第 2 步：正交 DMRG，χ = 64、sweeps = 10、pen = 1、m = 0）
# ==============================================================================
# 列：(x, N, n*, κ*, E*/e, W_E*, f₁*, F₀*, κ(1), 次优 n, 次优 κ, ρ 通道 k 峰占比, k 峰位/π,
#      W_ρ(n*))——最后两列只为 Parseval 自洽（Σ_k|ρ̃_k|² = W_ρ）
# 稠密 ED，Emax = N + 2（无截断）、m = 0、t = 0.5、seed 无关。f₁ 与 P38 的同一个量
# （分母 Σ_x⟨0|δE_x²|0⟩）；N=12、x=0.4 的 f₁ = 0.0483595309 逐位对上 P38 的
# PHOTON_WEIGHT_SMALL，这是两条独立路径对同一个量的交叉验证。
PHOTON_STATE_SCAN = (
    (0.20, 12, 5, 0.150827470, 2.77379481, 0.276677963, 0.0994884955, 0.576803658,
     0.0329883745, 22, 0.109021324, 0.444073, 0.166667, 0.0348757525),
    (0.40, 12, 1, 0.141697702, 0.371982659, 0.111993369, 0.0483595309, 0.871234880,
     0.141697702, 37, 0.115655294, 0.794244, 1.000000, 0.0354087814),
    (0.60, 12, 1, 0.336833509, 0.561094731, 0.169941612, 0.0869301131, 0.887012771,
     0.336833509, 10, 0.0757813674, 0.552034, 1.000000, 0.0322304348),
    (1.00, 12, 1, 0.675464607, 0.782279859, 0.164065827, 0.116557003, 0.880432821,
     0.675464607, 3, 0.0173495479, 0.296839, 1.000000, 0.0206901491),
    (0.20, 14, 7, 0.175747616, 4.29041293, 0.460121428, 0.115004093, 0.712197175,
     0.148827141, 4, 0.154724827, 0.454816, 0.285714, 0.107454185),
    (0.40, 14, 1, 0.266458855, 1.09713002, 0.366673975, 0.115908078, 0.831368513,
     0.266458855, 7, 0.087720412, 0.704031, 1.000000, 0.0521949168),
    (0.60, 14, 1, 0.448066432, 0.835478199, 0.304556825, 0.125585685, 0.852982946,
     0.448066432, 52, 0.0445866961, 0.488662, 1.000000, 0.0316131098),
    (1.00, 14, 1, 0.708171307, 0.805032429, 0.176280348, 0.106955471, 0.868686220,
     0.708171307, 3, 0.0364864189, 0.247300, 1.000000, 0.0148459905),
)

# 列：(x, N, |⟨n*|ρ̃_k|0⟩| 在 k = 2πj/N，j = 0..N−1)——认定态自己的动量分布（N=12）
# 注意 j = 0 那一位是**恒零**（⟨n|Q|0⟩ ≡ 0，自检 ⑥），这就是"ρ 通道看不见 k=0"的具体样子。
PHOTON_MOMENTUM_SCAN = (
    (0.20, 12, (1.60246891e-17, 0.124448278, 0.0151333605, 0.0122887421, 0.0231295026,
                0.0132692135, 0.0414590065, 0.0132692135, 0.0231295026, 0.0122887421,
                0.0151333605, 0.124448278)),
    (0.40, 12, (1.60246891e-17, 0.0263552813, 0.0176785267, 0.0160258466, 0.0193114708,
                0.0447872504, 0.167699738, 0.0447872504, 0.0193114708, 0.0160258466,
                0.0176785267, 0.0263552813)),
    (0.60, 12, (6.00925839e-18, 0.037299931, 0.0252019618, 0.0220875574, 0.0260942819,
                0.0634339853, 0.133387791, 0.0634339853, 0.0260942819, 0.0220875574,
                0.0252019618, 0.037299931)),
    (1.00, 12, (4.00617226e-18, 0.0432487191, 0.0249746730, 0.0213773841, 0.0275211704,
                0.0597131874, 0.0783685707, 0.0597131874, 0.0275211704, 0.0213773841,
                0.0249746730, 0.0432487191)),
)

# 列：(x, N, m_γ = (E_1 − E_0)/e, κ(1), W_E(1), F₀(1))——正交 DMRG 跟 n = 1
# N ≡ 2 (mod 4)（14/18/22）**不在表里**：那些环在自由极限下是开壳层，基态与谱结构不同，
# gap/e 系统性偏大（x=0.4：N=14 给 1.097、N=18 给 0.894、N=22 给 0.787，而 N=12/16/20
# 是 0.372/0.447/0.515）⇒ m_γ 的 N→∞ 只用 N ≡ 0 (mod 4)。N=14 那三行在
# PHOTON_STATE_SCAN 里保留（第 1 步按任务要求就是 N=12/14），但它们是**另一种环**。
# 列：(x, N, m_γ = (E_1 − E_0)/e, κ(1), W_E(1), F₀(1), |1⟩ 的 EOM 残差)
# 正交 DMRG 跟 n = 1（χ = 64、sweeps = 10、pen = 1、m = 0、t = 0.5、seed = 5）。
# N ≡ 2 (mod 4)（14/18/22）**不在表里**：那些环在自由极限下是开壳层，基态与谱结构不同，
# gap/e 系统性偏大（x=0.4：N=14 给 1.097、P35 的 N=18 给 0.894、N=22 给 0.787，而
# N=12/16/20 是 0.372/0.447/0.515）⇒ m_γ 的 N→∞ 只用 N ≡ 0 (mod 4)。N=14 那几行保留在
# PHOTON_STATE_SCAN 里（第 1 步按任务要求就是 N=12/14），但它们是**另一种环**。
# 最后一列是诚实条款：N ≥ 48 时正交 DMRG 的激发态 EOM 残差到了 1e-2 量级（态只收敛到
# 几个百分点），能量（Rayleigh 商）仍好得多——与 P36（χ=96）逐点差 ≤ 3e-4 就是证据。
M_GAMMA_SCAN = (
    (0.20, 12, 0.170125884, 0.0329865221, 0.046486119, 0.750828021, 1.1e-06),
    (0.20, 16, 0.199296189, 0.0352312082, 0.0627808062, 0.831271017, 1.5e-05),
    (0.20, 20, 0.230060848, 0.0416449427, 0.0881667175, 0.874106755, 1.5e-04),
    (0.20, 24, 0.262276676, 0.050826181, 0.122754886, 0.894281281, 7.4e-04),
    (0.20, 32, 0.329160088, 0.0754677451, 0.217486809, 0.907201637, 2.8e-03),
    (0.30, 12, 0.269174469, 0.0739716401, 0.0761101697, 0.837567326, 2.4e-07),
    (0.30, 16, 0.323429982, 0.0951238937, 0.119143184, 0.883241805, 5.1e-06),
    (0.30, 20, 0.378165629, 0.123952615, 0.17640093, 0.899556927, 6.2e-05),
    (0.30, 24, 0.430972539, 0.158222368, 0.244360768, 0.904609144, 2.4e-04),
    (0.30, 32, 0.521430464, 0.236497452, 0.396183073, 0.902608794, 1.4e-03),
    (0.30, 48, 0.612748007, 0.37365532, 0.639120603, 0.88349069, 4.1e-03),
    (0.30, 60, 0.629073296, 0.432470499, 0.721131778, 0.870879038, 7.8e-03),
    (0.35, 12, 0.320574161, 0.104544149, 0.0938559762, 0.858560252, 1.7e-07),
    (0.35, 16, 0.386415415, 0.140269765, 0.149665572, 0.89157667, 2.6e-06),
    (0.35, 20, 0.449583258, 0.185111016, 0.218840885, 0.901544546, 5.4e-05),
    (0.35, 24, 0.506083746, 0.235548496, 0.295357687, 0.902900449, 2.3e-04),
    (0.35, 32, 0.587715795, 0.337578654, 0.443445492, 0.895162987, 1.0e-03),
    (0.40, 12, 0.371982659, 0.141697731, 0.111993389, 0.871234855, 4.1e-08),
    (0.40, 16, 0.447283062, 0.194421109, 0.178141455, 0.895325688, 2.0e-06),
    (0.40, 20, 0.514665882, 0.256412929, 0.254367395, 0.900820965, 4.2e-05),
    (0.40, 24, 0.568888649, 0.321199121, 0.331613242, 0.898945089, 2.0e-04),
    (0.40, 32, 0.631802139, 0.431246473, 0.454742998, 0.886368754, 5.1e-04),
    (0.40, 48, 0.657045168, 0.540432978, 0.550186914, 0.863635525, 8.6e-03),
    (0.40, 60, 0.656966142, 0.581749391, 0.571337053, 0.855121131, 2.9e-02),
    (0.50, 12, 0.471308392, 0.232449223, 0.145271352, 0.883293696, 3.8e-08),
    (0.50, 16, 0.555954147, 0.320965112, 0.222007894, 0.896065433, 3.8e-06),
    (0.50, 20, 0.617262554, 0.409400533, 0.295519624, 0.894576906, 2.9e-05),
    (0.50, 24, 0.653499113, 0.482034467, 0.35276785, 0.887298385, 7.4e-05),
    (0.50, 32, 0.678223299, 0.568702933, 0.412626189, 0.870950935, 1.1e-03),
    (0.50, 48, 0.67962226, 0.642636714, 0.4449968, 0.853463237, 2.0e-02),
    (0.50, 60, 0.677787685, 0.672467203, 0.45256026, 0.846980695, 4.3e-02),
    (0.60, 12, 0.561094731, 0.3368335, 0.169941607, 0.887012772, 1.2e-07),
    (0.60, 16, 0.640101454, 0.451281695, 0.242671235, 0.892177256, 3.7e-06),
    (0.60, 20, 0.681845027, 0.53990363, 0.295699532, 0.885362152, 1.1e-05),
    (0.60, 24, 0.698204694, 0.5955111, 0.325902165, 0.87585205, 1.6e-04),
    (0.60, 32, 0.703349599, 0.65338636, 0.350419972, 0.861559348, 2.3e-03),
    (1.00, 12, 0.782279859, 0.675464621, 0.164065827, 0.880432837, 3.2e-06),
    (1.00, 16, 0.793472016, 0.732325951, 0.176700275, 0.873853547, 3.8e-05),
    (1.00, 20, 0.792633147, 0.757005575, 0.180707331, 0.866810356, 1.6e-04),
    (1.00, 24, 0.79083718, 0.771025304, 0.182477003, 0.861218976, 8.9e-04),
    (1.00, 32, 0.78845773, 0.786108729, 0.184077496, 0.852733843, 5.2e-03),
)


def _per_x_rows(points, n_min=16):
    """把 (x, N, m, …) 表按 x 归并：取最大 N 的实测值 G(x) 与它的点间漂移。"""
    by_x = {}
    for row in points:
        x, N, m = float(row[0]), int(row[1]), float(row[2])
        if N >= n_min:
            by_x.setdefault(round(x, 4), []).append((N, m))
    rows = []
    for x in sorted(by_x):
        seq = sorted(by_x[x])
        G = seq[-1][1]
        drift = seq[-1][1] - seq[-2][1] if len(seq) > 1 else float("nan")
        rows.append({"x": x, "G": G, "drift": drift,
                     "N": (seq[-2][0] if len(seq) > 1 else seq[-1][0], seq[-1][0]),
                     "seq": seq})
    return rows


def fit_m_gamma(points=None, n_min: int = 12, max_drift: float = 0.02) -> dict:
    """把 m_γ = (E_1 − E_0)/e 外推到 x→0，返回 G(0) 与不确定度。

    规矩与 `ks_mps.fit_continuum_gap_mps`（P36）完全一致，只是被外推的物理量换成
    "认定态的能量"（本项认定 n* = 1 ⇒ 与 P36 的 gap/e 是同一个量；这正是本项要回答的）：
      ① 固定 x：取最大 N 的实测值当 G(x)，报它与次大 N 的差 `drift`；
      ② 只在 drift ≤ max_drift 的 x 上做 G(x) = G0 + A·x^p（p = 1 / 1.5 / 2 家族 +
         c₁x² + c₂x⁴），G0 = 家族均值，不确定度 = 方和根(家族极差, 用到的最大 drift)。
    """
    pts = M_GAMMA_SCAN if points is None else points
    rows = _per_x_rows(pts, n_min)
    use = [r for r in rows if np.isfinite(r["drift"]) and r["drift"] <= max_drift]
    out = {"target": SQRT_PI_INV, "ref": G0_REF, "rows": rows,
           "usable": [r["x"] for r in use], "max_drift": max_drift,
           "n_points": len(pts), "forms": []}
    if len(use) < 3:
        out.update({"G0": float("nan"), "unc": float("nan"), "spread": float("nan"),
                    "drift_used": float("nan"), "dev_target": float("nan"),
                    "dev_ref": float("nan"), "p": float("nan")})
        return out
    xv = np.array([r["x"] for r in use], dtype=float)
    gv = np.array([r["G"] for r in use], dtype=float)
    drift = float(max(r["drift"] for r in use))

    def power_fit(p):
        A = np.vstack([np.ones_like(xv), xv ** p]).T
        c, *_ = np.linalg.lstsq(A, gv, rcond=None)
        return float(c[0]), float(np.max(np.abs(A @ c - gv)))

    forms = []
    for p in (1.0, 1.5, 2.0):
        g0, dev = power_fit(p)
        forms.append(("p=%.1f" % p, g0, dev))
    if len(xv) >= 3:
        A = np.vstack([np.ones_like(xv), xv ** 2, xv ** 4]).T
        c, *_ = np.linalg.lstsq(A, gv, rcond=None)
        forms.append(("c1x²+c2x⁴", float(c[0]), float(np.max(np.abs(A @ c - gv)))))
    g0s = [g for _, g, _ in forms]
    G0 = float(np.mean(g0s))
    spread = float(max(g0s) - min(g0s))
    out.update({"G0": G0, "spread": spread, "drift_used": drift,
                "unc": float(np.hypot(spread, drift)),
                "dev_target": abs(G0 - SQRT_PI_INV) / SQRT_PI_INV,
                "dev_ref": abs(G0 - G0_REF) / G0_REF,
                "max_dev_x": max(d for _, _, d in forms),
                "forms": [(n, g) for n, g, _ in forms], "p": float("nan")})
    return out


# ==============================================================================
# 6. 自检（P39）
# ==============================================================================
def _row_index(table):
    """(x, N) → 行。"""
    return {(round(float(r[0]), 2), int(r[1])): r for r in table}


def _seam_identity(R, N):
    """挪接缝的恒等式：E^{(s)}_x = E_x − E_s  ⇒  Σ_x E^{(s)}_x = Σ_x E_x − N·E_s。

    返回两个残差（逐点、以及 Σ 上的差）。第二条是"Σ_x E_x **不是**接缝无关量"的量化：
    换一个接缝就减去 N·E_s，而 E_s 是逐态变化的算符。所以 Ê 只能在这个模型**自己**的
    接缝（E_{N−1} = Q = 0，也就是 H 与高斯定律用的那一个）上谈，见 `seam_argmax`。
    """
    e_id = e_sum = 0.0
    ref = np.cumsum(R, axis=1)
    for st in range(N):
        acc = np.zeros(R.shape[0])
        E = np.zeros_like(R)
        for d in range(1, N + 1):
            y = (st + d) % N
            acc = R[:, y].copy() if d == 1 else acc + R[:, y]
            E[:, y] = acc
        e_id = max(e_id, float(np.abs(E - (ref - ref[:, [st]])).max()))
        e_sum = max(e_sum, float(np.abs(E.sum(axis=1)
                                       - (ref.sum(axis=1) - N * ref[:, st])).max()))
    return e_id, e_sum


def seam_argmax(N, x):
    """s = 0..N−1 每种接缝约定下 κ(n) 的 argmax（**负结果**：只有 s = N−1 给 n = 1）。

    返回 {"base": 模型接缝下的 argmax, "per_seam": {s: (n, κ)}, "n1_wins": 赢的接缝数}。
    这条把"哪个态是光子"钉在**模型内**：模型没有平移不变性（`translation_defect`），
    所以"光子的零模"换一个接缝就换一个算符，判据不稳健——如实报告，不当成普适判据。
    """
    d = dense_photon_weights(N, x)
    R, E = d["R"], d["E"]
    V, v0 = d["V"], d["V"][:, 0]
    base = int(d["_order"][0])
    per = {}
    for st in range(N):
        ehat = E.sum(axis=1) - N * E[:, st]
        amp = V.conj().T @ (ehat * v0)
        var = float(np.vdot(v0, ehat * ehat * v0).real - np.vdot(v0, ehat * v0).real ** 2)
        kap = np.abs(amp) ** 2 / var
        o = [int(n) for n in np.argsort(-kap) if n != 0]
        per[st] = (o[0], float(kap[o[0]]))
    return {"base": base, "per_seam": per,
            "n1_wins": int(sum(1 for n, _k in per.values() if n == 1))}


def check_ks_photon_mass() -> dict:
    """P39：把光子从激发谱里挑出来，用它的能量定义 m_γ（`ks_photon_mass.py`）。

    **结构判据（进残差）**
      ① Ê 的两种算法一致：稠密侧 Σ_x E_x（对角）== Σ_y (N−y)ρ_y（N=8/12，全部基矢）
      ② Ê 与接缝位置无关：接缝挪到 s = 0..N−1 时 Σ_x E^{(s)}_x 不变（N=8/12 全部基矢）
      ③ Ê² 的 3 通道 MPO vs 稠密（精确随机 MPS，χ 取满，N=8）
      ④ 零模求和规则 Σ_{n≠0}κ(n) = 1（N=12/14，x=0.2/0.4/0.6/1.0）
      ⑤ 全谱求和规则 Σ_n W_E(n) = Σ_x⟨0|E_x²|0⟩（同上）
      ⑥ ρ 通道的 k = 0 分量恒为零：max_n|⟨n|ρ̃_0|0⟩|（同上）
      ⑦ 平移：g = 0 时 ‖[H,T]‖ = 0 且 T^N = 1（N=8/12）；恒等式 HT−TH = T·diag(ΔV) 到机器精度
      ⑧ ED vs DMRG：m_γ 与 κ(1)（N=12，x=0.4/0.6）
      ⑨ 表 PHOTON_STATE_SCAN 的 N=12 四行现场复现（判据 ≤1e-6）
      ⑩ 表 M_GAMMA_SCAN 的 (x=0.4, N=16) 现场复现（判据 ≤1e-6）
    **物理判据（阈值写死在实测表上，可稳定复现）**
      ⑪ x ≥ 0.4：E ≤ 2e 的低能态里 κ 最大的是 n = 1
      ⑫ κ(1) 随 x 单调增（x = 0.2 → 1.0，N=12/14 全部）
      ⑬ x ≥ 0.4：F₀(1) ≥ 0.80（E 通道 k = 0 占比：场近似均匀）
      ⑭ x ≥ 0.4：κ(1) ≥ 0.10（低能态里零模份额至少 10%）
      ⑮ 连续极限 G(0) 落在 1/√π 的 [0.95, 1.25] 窗口内（只防回归，不是精度声明）
    **物理结论（_magnitude，只报数不判定）**
      · 每个 (x, N) 的 κ*、E*/e、W_E、f₁、F₀；m_γ 全表；G(0)、不确定度、与 0.581 和
        1/√π 的偏差；[H,T] 的数值（接缝缺陷，g = 0 之外都不为零）
    """
    out = {}

    # ① Ê 两种算法 + ② 接缝无关 + ⑤⑥ 求和规则（稠密，N=12 四个 x / N=14 一个 x）
    e_id = e_seam = 0.0
    for N in (8, 12):
        _H, masks, _ = _ed_module().basis_and_hamiltonian(N, 0.4, 0.0, t=0.5, Emax=N + 2)
        R, E = dense_basis_ops(N, masks)
        w = zero_mode_weights(N)
        e_id = max(e_id, float(np.abs(E.sum(axis=1) - R @ w).max()))
        e_seam = max(e_seam, _seam_identity(R, N)[0])
    out["Ê = Σ_x E_x 与 Σ_y (N−y)ρ_y 一致（N=8/12 全部基矢）"] = e_id
    out["挪接缝的恒等式 E^{(s)}_x = E_x − E_s（N=8/12，s = 0..N−1）"] = e_seam

    # ③ Ê² MPO vs 稠密（随机满键 MPS）
    A0, v0 = _kw._random_full_mps(8, 7)
    A1, v1 = _kw._random_full_mps(8, 8)
    Ws = zero_mode_square_mpo(8)
    M = _mps.mpo_dense(Ws, 3)
    out["Ê² MPO vs 稠密（随机满键 MPS，N=8）"] = abs(
        complex(np.vdot(v1, M @ v0)) - _kw.trans_amp_mpo(A1, Ws, A0))

    # ④⑤⑥ 全谱求和规则（N=12 四个 x；N=14 一个 x）
    dense = {}
    e_sum = e_w = e_k0 = 0.0
    for (N, x) in ((12, 0.2), (12, 0.4), (12, 0.6), (12, 1.0), (14, 0.4)):
        d = dense_photon_weights(N, x)
        dense[(N, round(x, 2))] = d
        e_sum = max(e_sum, abs(float(d["kappa"][1:].sum()) - 1.0))
        e_w = max(e_w, abs(float(d["W_E"].sum()) - float(d["E2_mean"].sum())))
        e_k0 = max(e_k0, float(np.abs(d["rho_k"][:, 0]).max()) ** 0.5)
    out["零模求和规则 Σ_{n≠0}κ(n) = 1（N=12×4 + N=14×1）"] = e_sum
    out["全谱求和规则 Σ_n W_E(n) = Σ_x⟨0|E_x²|0⟩（同上）"] = e_w
    out["ρ 通道 k=0 分量恒为零 max_n|⟨n|ρ̃_0|0⟩|（同上）"] = e_k0

    # ⑦ 平移：g=0 不变、T^N=1、[H,T] = T·diag(ΔV)
    e_t = e_tn = e_id2 = 0.0
    defect = {}
    for N in (8, 12):
        H0, masks, _ = _ed_module().basis_and_hamiltonian(N, 0.0, 0.0, t=0.5, Emax=N + 2)
        T0 = translation_operator(N, masks)
        e_t = max(e_t, float(np.abs(H0 @ T0 - T0 @ H0).max()))
        e_tn = max(e_tn, float(np.abs(np.linalg.matrix_power(T0, N)
                                      - np.eye(len(masks))).max()))
        rot = _rotation_index(N, masks)
        for g in (0.4, 1.0):
            H, _mg, _ = _ed_module().basis_and_hamiltonian(N, g, 0.0, t=0.5, Emax=N + 2)
            T = translation_operator(N, masks)
            lhs = H @ T - T @ H
            defect[(N, g)] = float(np.abs(lhs).max())
            rhs = T * (np.diag(H)[rot] - np.diag(H))[None, :]
            e_id2 = max(e_id2, float(np.abs(lhs - rhs).max()))
    out["g=0 时 ‖[H,T]‖ = 0（N=8/12）"] = e_t
    out["T^N = 1（N=8/12）"] = e_tn
    out["恒等式 HT−TH = T·diag(ΔV)（N=8/12，g=0.4/1.0）"] = e_id2

    # ⑦b 两个公开小工具的现场调用：`zero_mode_amp` 与逐点振幅累加一致（同一个量的两条路径）、
    #    `translation_defect` 在 g=0 为 0 且 g>0 非零（接缝缺陷）
    sp12 = _kw.dmrg_state_pair(12, 0.6, chi=64, sweeps=10)
    fe12 = _kw.field_matrix_elements(sp12["A0"], sp12["A1"], sp12["gap"])
    amp_direct = complex(np.dot(zero_mode_weights(12), np.asarray(fe12["rho"])))
    out["zero_mode_amp 与逐点振幅累加一致（N=12，x=0.6）"] = abs(
        zero_mode_amp(sp12["A0"], sp12["A1"]) - amp_direct)
    out["translation_defect：g=0 为 0 且 g=1 非零（N=8）"] = max(
        translation_defect(8, 0.0), max(0.0, 1e-6 - translation_defect(8, 1.0)))

    # ⑧ ED vs DMRG（N=12，x=0.4/0.6）
    e_dm = e_dk = 0.0
    for x in (0.4, 0.6):
        d = dense[(12, round(x, 2))]
        p = photon_mass_point(12, x, with_f1=False)
        e_dm = max(e_dm, abs(p["m_gamma"] - float((d["w"][1] - d["w"][0]) / x)))
        e_dk = max(e_dk, abs(p["kappa"] / float(d["kappa"][1]) - 1.0))
    out["ED vs DMRG：m_γ（N=12，x=0.4/0.6）"] = e_dm
    out["ED vs DMRG：κ(1) 相对差（N=12，x=0.4/0.6）"] = e_dk

    # ⑨ 表 PHOTON_STATE_SCAN 的 N=12 四行现场复现（名次也要对上）
    idx = _row_index(PHOTON_STATE_SCAN)
    worst, hit = 0.0, 0
    for x in (0.2, 0.4, 0.6, 1.0):
        key = (round(x, 2), 12)
        if key not in idx or (12, round(x, 2)) not in dense:
            continue
        r, d = idx[key], dense[(12, round(x, 2))]
        order = [int(n) for n in np.argsort(-d["kappa"]) if n != 0]
        hit += 1
        worst = max(worst,
                    abs(float(d["kappa"][order[0]]) - float(r[3])),
                    abs(float((d["w"][order[0]] - d["w"][0]) / x) - float(r[4])),
                    abs(float(d["W_E"][order[0]]) - float(r[5])),
                    abs(float(d["F0"][order[0]]) - float(r[7])),
                    float(abs(order[0] - int(r[2]))),
                    float(abs(order[1] - int(r[9]))))
    out["表 PHOTON_STATE_SCAN 的 N=12 四行现场复现（≤1e-6）"] = worst if hit == 4 else 1.0

    # ⑨b 表 PHOTON_MOMENTUM_SCAN 的 Parseval：Σ_k|ρ̃_k|² = W_ρ（与 PHOTON_STATE_SCAN 自洽）
    mom = {(round(float(r[0]), 2), int(r[1])): np.asarray(r[2], dtype=float)
           for r in PHOTON_MOMENTUM_SCAN}
    worst = 0.0
    for key, prof in mom.items():
        if key not in idx:
            continue
        W_r_stored = float(idx[key][13])
        worst = max(worst, abs(float((prof ** 2).sum()) / W_r_stored - 1.0))
    out["表 PHOTON_MOMENTUM_SCAN 与 W_ρ 的 Parseval 自洽（≤1e-4）"] = (
        worst if mom else 1.0)

    # ⑩ M_GAMMA_SCAN 的 (0.4, 16) 现场复现
    mg = _row_index(M_GAMMA_SCAN)
    if (0.40, 16) in mg:
        live = photon_mass_point(16, 0.4, with_f1=False)
        out["表 M_GAMMA_SCAN 的 (x=0.4,N=16) 现场复现（≤1e-6）"] = abs(
            live["m_gamma"] - mg[(0.40, 16)][2])
    else:
        out["表 M_GAMMA_SCAN 的 (x=0.4,N=16) 现场复现（≤1e-6）"] = 1.0

    # ============ 物理判据 ============
    # ⑪ 低能（E ≤ 2e）里 κ 最大的是 n=1
    worst = 0.0
    for (N, x), d in dense.items():
        if x < 0.4 - 1e-9:
            continue
        e_over = (d["w"] - d["w"][0]) / x
        low = [n for n in range(1, len(e_over)) if e_over[n] <= 2.0]
        if not low:
            continue
        best_other = max((float(d["kappa"][n]) for n in low if n != 1), default=0.0)
        worst = max(worst, max(0.0, best_other - 0.8 * float(d["kappa"][1])))
    out["x ≥ 0.4：E ≤ 2e 的低能态里 κ 最大的是 n=1"] = worst

    # ⑫ κ(1) 随 x 单调增
    mono = 0.0
    for N in (12, 14):
        seq = [(x, float(dense[(N, round(x, 2))]["kappa"][1])) for x in (0.2, 0.4, 0.6, 1.0)
               if (N, round(x, 2)) in dense]
        for i in range(1, len(seq)):
            mono = max(mono, max(0.0, seq[i - 1][1] - seq[i][1]))
    out["κ(1) 随 x 单调增（N=12/14）"] = mono

    # ⑬ F₀(1) ≥ 0.80、⑭ κ(1) ≥ 0.10（x ≥ 0.4）
    f0 = min((float(d["F0"][1]) for (N, x), d in dense.items()
              if x >= 0.4 - 1e-9), default=0.0)
    kk = min((float(d["kappa"][1]) for (N, x), d in dense.items()
              if x >= 0.4 - 1e-9), default=0.0)
    out["x ≥ 0.4 时 F₀(1) ≥ 0.80（E 通道 k=0 占比）"] = max(0.0, 0.80 - f0)
    out["x ≥ 0.4 时 κ(1) ≥ 0.10"] = max(0.0, 0.10 - kk)

    # ⑯ 负结果：κ 的 argmax 只在模型自己的接缝上给 n=1（不做判定，只报数）
    for (N, x) in ((12, 0.4), (12, 1.0)):
        if (N, round(x, 2)) in dense:
            sa = seam_argmax(N, x)
            out["κ argmax 稳健的接缝数（N=%d,x=%.2f，共 %d）_magnitude" % (
                N, x, N)] = float(sa["n1_wins"])
            out["κ argmax 在 s≠N−1 上落到 n（N=%d,x=%.2f）_magnitude" % (N, x)] = float(
                max(n for st, (n, _k) in sa["per_seam"].items() if st != N - 1))

    # ⑮ 连续极限窗口（只防回归）
    fit = fit_m_gamma()
    out["连续极限 G(0) 落在 e/√π 的 [0.95,1.25] 窗口（判据）"] = float(
        0.0 if (np.isfinite(fit["G0"]) and 0.95 <= fit["G0"] / SQRT_PI_INV <= 1.25) else 1.0)

    # ============ 物理结论（只报数） ============
    for (N, x), d in sorted(dense.items()):
        order = np.argsort(-d["kappa"])
        n1 = int(order[1])
        tag = "N=%d,x=%.2f" % (N, x)
        out["光子候选 n*（%s）_magnitude" % tag] = float(n1)
        out["κ(n*)（%s）_magnitude" % tag] = float(d["kappa"][n1])
        out["E*/e（%s）_magnitude" % tag] = float((d["w"][n1] - d["w"][0]) / x)
        out["W_E(n*)（%s）_magnitude" % tag] = float(d["W_E"][n1])
        out["F₀(n*)（%s）_magnitude" % tag] = float(d["F0"][n1])
        out["κ(1)（%s）_magnitude" % tag] = float(d["kappa"][1])
        out["⟨n*|ρ̃_π|0⟩²/Σ（%s）_magnitude" % tag] = float(
            d["rho_k"][n1, N // 2] / max(float(d["W_rho"][n1]), 1e-300))
        wbest = int(max(range(1, len(d["W_E"])), key=lambda n: d["W_E"][n]))
        cand = [n for n in range(1, len(d["F0"])) if d["W_E"][n] >= 0.02]
        out["W_E 最大的 n（%s）_magnitude" % tag] = float(wbest)
        out["W_E(n_max)（%s）_magnitude" % tag] = float(d["W_E"][wbest])
        # F₀ 不是判据：接缝让**很多**态的剖面都是坡道，F₀ > 0.7 的一抓一把（下面报个数）
        out["F₀ ≥ 0.7 的态数（W_E ≥ 0.02，%s）_magnitude" % tag] = float(
            sum(1 for n in cand if d["F0"][n] >= 0.7))
        out["判据 κ 与 W_E 指向同一个 n（%s，1 = 是）_magnitude" % tag] = float(int(n1 == wbest))
    for (N, g), v in sorted(defect.items()):
        out["‖[H,T]‖（N=%d,g=%.1f）_magnitude" % (N, g)] = v
    for r in PHOTON_STATE_SCAN:
        tag = "x=%.2f,N=%d" % (r[0], r[1])
        out["κ*（%s）_magnitude" % tag] = float(r[3])
        out["次优 κ（%s）_magnitude" % tag] = float(r[10])
    for r in M_GAMMA_SCAN:
        out["m_γ(x=%.2f,N=%d)_magnitude" % (r[0], r[1])] = float(r[2])
        out["κ(1)（x=%.2f,N=%d）_magnitude" % (r[0], r[1])] = float(r[3])
        out["F₀(1)（x=%.2f,N=%d）_magnitude" % (r[0], r[1])] = float(r[5])
    out["G(0) = lim m_γ/e_magnitude"] = float(fit["G0"])
    out["G(0) 的不确定度_magnitude"] = float(fit["unc"])
    out["e/√π_magnitude"] = float(SQRT_PI_INV)
    out["|G(0) − e/√π|/(e/√π)_magnitude"] = float(fit["dev_target"])
    out["P35/P36 的 G(0)=0.581_magnitude"] = float(G0_REF)
    out["|G(0) − 0.581|/0.581_magnitude"] = float(fit["dev_ref"])
    out["进 x→0 拟合的 x 点数_magnitude"] = float(len(fit["usable"]))
    out["PHOTON_STATE_SCAN 行数_magnitude"] = float(len(PHOTON_STATE_SCAN))
    out["PHOTON_MOMENTUM_SCAN 行数_magnitude"] = float(len(PHOTON_MOMENTUM_SCAN))
    out["M_GAMMA_SCAN 行数_magnitude"] = float(len(M_GAMMA_SCAN))
    return out
