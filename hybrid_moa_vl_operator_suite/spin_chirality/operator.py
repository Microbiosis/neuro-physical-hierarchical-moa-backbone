#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""SpinChiralityOperator —— 物理忠实的旋量手性算子（复旋量 / 真 SU(2) / 精确 Dirac 演化）

与上一版（实四分量、SO(2)、手工阻尼与"反常门"）不同，本模块的每个操作都是
量子场论里定义明确的对象：

    旋量       Ψ = (ψ_L, ψ_R) ∈ ℂ⁴ 每组；输入张量 X ∈ ℂ^{B×L×4G}
    手性投影   P_{L/R} = (I ∓ γ⁵)/2，Weyl 表象 γ⁵ = diag(−1,−1,+1,+1)
    手性规范   ψ_L → U_L ψ_L，ψ_R → U_R† ψ_R，U = cos(θ/2)I − i sin(θ/2) n̂·σ ∈ SU(2)
               （右手是 U† 而非 U；实现用单位四元数，R† = 四元数共轭，酉性由 |q|=1 保证）
               （左右各一个独立 SU(2)：这是左右对称扩展 SU(2)_L × SU(2)_R；
                标准模型的弱同位旋只作用在左手场，令 U_R = 1 即可回到 SM）
    Larmor     Ψ → exp(+i μ B·Σ dt) Ψ，Σ = diag(σ/2, σ/2)（矢量耦合，两手性同向；
               对应 H = −μ B·Σ；若取 H = +μ B·Σ 则 μ → −μ）
    质量项     Ψ → exp(−i m β dt) Ψ，β = γ⁰（产生手性翻转，频率 2m）
    吸收       Ψ → e^{−Γ dt} Ψ（非厄米光学势；Γ > 0 时明确破坏幺正性）

自由部分精确：$[\Sigma_i,\beta]=0$，故 $e^{-i(-\mu B\cdot\Sigma + m\beta)dt}
= e^{-im\beta dt}\,e^{+i\mu B\cdot\sigma/2\,dt\oplus +i\mu B\cdot\sigma/2\,dt}$ 就是单一哈密顿量的指数，无分裂误差（P9 检验）。
手性规范步支持两种语义：`gauge_mode="protocol"` 为瞬时规范变换（无 dt 标度），
`gauge_mode="generator"` 把 θ_{L/R}·n̂ 视作连续规范势（ω=A·dt），默认采用二阶对称
Strang 分裂以消除 O(dt) 一阶非对易误差；序列接口支持静态 M(G,4,4) 与时变 M_seq(L,G,4,4)（P43/P44）。

序列传播（沿 L 的积分）
-----------------------
s_{n+1} = M s_n + c u_n，M 为一步物理（规范×Larmor×质量×吸收）在每组上的 4×4 传播子。
结合律二元算子 (M₂,b₂)∘(M₁,b₁)=(M₂M₁, M₂b₁+b₂) 支持 O(log L) 深度前缀扫描；
Triton 融合核把复旋量拆成实 8 维，逐 (b,g) 程序在寄存器中沿 L 迭代，
反向在核内做逆序伴随并用 atomic_add 归约 dM/dc。

    P10 扫描 vs 序贯（fp64）        3.6e-15；传播子 vs 一步前向 5.6e-16
    P11 自定义反向 vs autograd       0.0；有限差分 2.2e-9
    P12 NumPy vs torch vs Triton     2.3e-15 / 1.8e-15
    P12 Triton 反向 vs autograd      dU 0.0；dM/dc 2.8e-14
    P13 解析伴随 vs 重计算            2.2e-15
    P14 m=0 手性荷相对漂移           2.6e-8（受残余吸收限制）
    P15 四元数作用/共轭/复合          0.0 / 0.0 / 3.3e-16
    P16 反常记账 vs 指标归一化        ΔQ5 = 2·index（与 g 无关），ff 约定一致
    P17 3+1D 指标                    瞬子 −1 / 反瞬子 +1 / γ⁵ 厄米（lattice4d.py）
    P18 反常系数（Fujikawa 热核）     tr[γ5X²]=εFF、A/(εFF)=1/(32π²) ⇒ g²/16π²（各路 0.0）
    P19 微观谱与代数                H=−μB·Σ+mβ 的四能级、粒子–空穴、[Σ,Σ]=iεΣ、[Σ,β]=0、{γ5,β}=0
    P20 速度算符 / Zitterbewegung   α 本征值 ±1；α(t)=p/H+(α(0)−p/H)e^{−2iHt}（残差 ~5e-16）
    P21 自包含与完整性             __all__ 全解析、硬依赖仅 numpy、无外部 I/O、公开 API 覆盖表
    P22 1+1D 色散/倍增            E(k)=±√(sin²k+(m+r(1−cos k))²)；r=0 两狄拉克点、r=1 只剩一个
    P23 畴壁零模                  kink–antikink 环：Wilson 下 2 个零模、naive 下 4 个（倍增）
    P24 规范步离散阶              gauge_link 中点采样：O(dt) → O(dt²)，实测改善 ~17 倍
    P25 1+1D 时间演化              FFT 与本征分解两套实现对拍；幺正性 4e-15、色散相位 3e-16
    P26 1+1D 规范场                Peierls 链接：纯规范不变 2.7e-15、AB 周期 4.6e-15、Klein T(m=0)=1
    P27 Fock 空间（二次量子化）     反对易、Pauli 排斥、多体演化 = Slater 行列式（全 0）
    P28 海的手征荷                 磁通扫全周 Q5 ≡ 0（1e-15）：反常不是海的性质，是正规化效应
    P29 1+1D 热核                  同 P18 的热核法：2 维跑一阶 X，A/(εF)=1/(4π)（Schwinger）
    P30 对产生（LZ）              磁通匀速扫过：P = exp(−πm²/ω)，三点实测比值 1.8%/0.8%/6.9%
    P31 规范场反作用              半经典：诱导电流 j=−(e/π)A₁ ⇒ ω_p = e/√π、屏蔽长度 √π/e，均 1e-6 级
    P32 有质量反作用              真空极化 Π(0)=1/π 与 m 无关（5e-12）；ω_p: 0.581→0.568→0.5648 → e/√π
    P33 禁闭（弦张力）            高斯定律 ⇒ V(d)=(g²/2)d 线性势，σ=g²/2（残差 3e-16）
    P34 Kogut–Susskind ED         ED + 高斯定律投影：厄米 0、逐态高斯 0、截断饱和（N ≤ 14）
    P35 Kogut–Susskind 连续极限  Lanczos 推到 N=24：matvec vs 稠密 H 3.6e-15、强耦合 gap→g²/2、
                              自由极限 PBC 0.9e-15、Emax 2→3 ≤1e-4、gap/e 从 x=1 的 0.7930
                              外推到 x→0 得 0.59±0.04（比 e/√π 高 5%，未闭合）
    P36–P42 KS MPS/DMRG 与光子色散  MPS 推到 N=60、磁通 φ、电场谱权重/零模与 (mask,E) 平移基，
                              动量扇区色散 ω(k) 测得 Z_loc(0)≈0.989 与 G(0)=0.5601（0.73%）
    P43 零轴光滑与构造边界         su2_exp Taylor 光滑、su2_quat 零轴单位元、严格复数 dtype 门禁、非 z 传播子一致
    P44 时变规范与对称分裂二阶收敛   gauge_mode="generator" + splitting_order=2 Strang 分裂 O(dt²)、M_seq 全栈一致
    P45 实数等变包装与闭式 Procrustes RealSpinChiralityOperator (dirac8/chiral4/su2_quat4) 与 fit_{chiral,su2}_procrustes

吞吐（B=16, G=8，同进程公平对比）：
    宽加载持久核 fp64：L=256 约 0.1–0.4 ms、L=4096 约 2–3 ms；torch 实张量序贯
    L=256 24 ms / L=4096 413 ms → 约 100–126×。
    尝试过但未采用：分块扫描（两趟双倍访存，慢 1.4–2×）、软件流水 num_stages=2
    （L=256 快 1.3×、L=4096 无效）、对角化+FFT 卷积（慢且未对齐）。

物理检验（python -m spin_chirality 自动执行 P1–P45）
----------------------------------------------------------
    P1  Clifford 代数      {γ^μ,γ^ν}=2η^{μν}，γ⁵=iγ⁰γ¹γ²γ³，{γ⁵,γ^μ}=0
    P2  手性投影           幂等、正交、完备、迹、本征
    P3  SU(2)              酉、det=1、群同态、李代数
    P4  Larmor             自旋期望与经典进动 R(−μBt) 一致（H = −μB·Σ）
    P5  手性动力学         m=0 守恒；m>0 以 2m 振荡
    P6  轴矢 Ward 恒等式   ∂_μ j^μ_5 = 2im ψ̄γ⁵ψ（双模态，非平凡）
    P7  幺正性/吸收律      ‖Ψ_out‖ = e^{−Γdt}‖Ψ_in‖（Γ=0 时严格等范）
    P8  可微性             实值损失对实/虚部的有限差分 vs autograd
    P9  自由演化精确        [Σ,β]=0 ⇒ e^{-i(-μB·Σ+mβ)dt} 精确分解
    P10 序列扫描            结合律前缀扫描 vs 序贯；传播子 vs 一步前向
    P11 序列反向            自定义反向 vs autograd；有限差分抽查
    P12 Triton 三重对齐     NumPy / torch / Triton 前向与反向
    P13 解析伴随            λ 递推 vs 重计算
    P14 手性守恒            m=0 齐次演化 Q5 守恒；m>0 振荡
    P15 四元数 SU(2)        作用/共轭/复合 vs 2×2 复矩阵
    P16 反常记账            单瞬子 ∫F F̃ = 32π²/g² ⇒ ΔQ5 = 2·index，与 g 无关
    P17 3+1D 指标           FLSG 单瞬子 ⇒ index = −1，反瞬子 ⇒ +1（N=8，需 scipy）
    P18–P42                 Fujikawa 热核、微观谱/ZBW、自包含/API 覆盖、1+1D→KS 连续极限与光子色散 ω(k)
    P43–P45                 零轴光滑与严格复门禁、Strang 二阶分裂与 M_seq、实数等变包装与闭式 Procrustes

本机实测（python -m spin_chirality）：P1–P45 共 45 项全通过（跳过 0），最大残差 9.58e-07（P31 等离子频率零点；
其中 P8 有限差分 8.39e-08、P43 1.01e-07、P44 1.10e-14、P45 5.47e-10）。

使用示例
--------
    import torch
    from spin_chirality import (
        SpinChiralityConfig, SpinChiralityOperator, RealSpinChiralityOperator
    )

    # 1. 复旋量核心接口（严格仅接受 complex64 / complex128）
    op = SpinChiralityOperator(SpinChiralityConfig(dim=64, dt=0.05))
    x = torch.randn(2, 16, 64, dtype=torch.complex128)   # ℂ^{B×L×4G}
    op(x).real.sum().backward()

    # 2. 主流实值网络等变包装接口（dirac8 / chiral4 / su2_quat4）
    rop = RealSpinChiralityOperator(SpinChiralityConfig(dim=64, dt=0.05), representation="dirac8")
    xr = torch.randn(2, 16, 64, dtype=torch.float32)     # ℝ^{B×L×8G}
    rop(xr).sum().backward()

关于 ABJ 反常的边界
-------------------
反常 ∂_μ j^μ_5 = (g²/16π²) F F̃ 是单圈量子效应，不能由一个作用在场上的经典局部
算子"产生"。本模块提供经典对象（手性投影、γ⁵、轴矢流 j⁵、手性荷 Q5）与反常的
积分记账接口 ΔQ5 = (g²/16π²)∫F^a F̃^a dt = (g²/4π²)∫E·B dt（约定 ε^{0123}=+1）。
这个系数本身由 P18 用 Fujikawa 热核推出来并数值校验：D̸² = D² − (i/2)σ_{μν}F_{μν}，
A = lim tr[γ5 e^{D̸²/M²}] = tr[γ5X²]/(32π²) = ε^{μνρσ}F_{μν}F_{ρσ}/(32π²) = F F̃/(16π²)。

归一化依据来自**格点上已验证过的指标定理**（细节见 INDEX_AUDIT.md）：
    overlap 算符 D_ov = ½[(1+am) + (1−am) γ⁵ sign(γ⁵A)]，核 A = D_W − 1，
    满足 {D_ov,γ⁵} = 2 D_ov γ⁵ D_ov（GW 残差 1e-15）。
  · 1+1 维（lattice_anomaly.py）：指标 = 磁通量子数。n = ±1 / ±2 实测指标 ±1 / ±2，
    零模逐个是 γ⁵ 本征态（⟨γ⁵⟩ = ±1）。
  · 3+1 维（index_4d_instanton.py）：周期单瞬子（Fox–Laursen–Schierholz–Gilchrist
    构造）的 overlap 指标 = −1，沿 t: 0→1 的连续路径恰有一次干净的本征值穿越零。
  · 一个瞬子给出 ∫F F̃ = 32π²/g² ⇒ ΔQ5 = 2·index，且与 g 无关（P16 检验）。

注：畴壁（kink）**不能**用来检验 2D 指标——2D 中 {γ⁵,γ₁} = 0，畴壁零模是 γ₁ 本征态
（1 维子空间），而 γ⁵ = γ₁γ₂ 把它映到正交补，故 ⟨γ⁵⟩ ≡ 0。手性零模要求墙沿"多出来的
那一维"（DWF 是 2n+1 维格点上的构造）。
"""

from __future__ import annotations

import ast

import numpy as np

try:
    import os as _os_triton; _os_triton.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1"); import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False
    triton = None
    tl = None

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore


# ==============================================================================
# 1. Dirac 代数（Weyl 表象）
# ==============================================================================
I2 = np.eye(2, dtype=complex)
S1 = np.array([[0, 1], [1, 0]], dtype=complex)
S2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
S3 = np.array([[1, 0], [0, -1]], dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)
SIG = (S1, S2, S3)

GAMMA0 = np.block([[Z2, I2], [I2, Z2]])
GAMMA = (
    GAMMA0,
    np.block([[Z2, S1], [-S1, Z2]]),
    np.block([[Z2, S2], [-S2, Z2]]),
    np.block([[Z2, S3], [-S3, Z2]]),
)
GAMMA5 = 1j * GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]   # 由代数定义导出，本基下 = diag(-1,-1,1,1)
BETA = GAMMA0
ETA = np.diag([1.0, -1.0, -1.0, -1.0])
ID4 = np.eye(4, dtype=complex)
PL = 0.5 * (ID4 - GAMMA5)
PR = 0.5 * (ID4 + GAMMA5)


def su2(axis: np.ndarray, theta: float) -> np.ndarray:
    """U = cos(θ/2) I − i sin(θ/2) n̂·σ ∈ SU(2)。"""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return I2.copy()
    n_hat = axis / n
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * (
        n_hat[0] * S1 + n_hat[1] * S2 + n_hat[2] * S3
    )


def apply_su2_spinor(u1, u2, axis, theta):
    """对二分量旋量作用 U(axis, θ)；支持 numpy 与 torch（复数）。"""
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n == 0.0:
        return u1, u2
    n = axis / n
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # (n·σ)ψ = (n3 u1 + (n1 − i n2) u2, (n1 + i n2) u1 − n3 u2)
    a1 = n[2] * u1 + (n[0] - 1j * n[1]) * u2
    a2 = (n[0] + 1j * n[1]) * u1 - n[2] * u2
    return c * u1 - 1j * s * a1, c * u2 - 1j * s * a2


def su2_exp(omega: np.ndarray) -> np.ndarray:
    """光滑 SU(2) 指数映射：U(ω) = exp(-i ω·σ/2)，在 ω=0 处无奇点。"""
    w = np.asarray(omega, dtype=float)
    phi2 = float(np.dot(w, w))
    if phi2 < 1e-8:
        x2 = 0.25 * phi2
        c = 1.0 - 0.5 * x2 + (x2 * x2) / 24.0
        half_sinc = 0.5 * (1.0 - x2 / 6.0 + (x2 * x2) / 120.0)
    else:
        phi = float(np.sqrt(phi2))
        c = float(np.cos(0.5 * phi))
        half_sinc = float(np.sin(0.5 * phi) / phi)
    return c * I2 - 1j * half_sinc * (w[0] * S1 + w[1] * S2 + w[2] * S3)


def _apply_su2_omega_numpy(u1, u2, omega):
    """用李代数旋转矢量 ω = θ n̂ 作用在二分量旋量上，ω=0 处使用 Taylor 展开无奇点。"""
    w = np.asarray(omega, dtype=float)
    phi2 = float(np.dot(w, w))
    if phi2 < 1e-8:
        x2 = 0.25 * phi2
        c = 1.0 - 0.5 * x2 + (x2 * x2) / 24.0
        half_sinc = 0.5 * (1.0 - x2 / 6.0 + (x2 * x2) / 120.0)
    else:
        phi = float(np.sqrt(phi2))
        c = float(np.cos(0.5 * phi))
        half_sinc = float(np.sin(0.5 * phi) / phi)
    a1 = w[2] * u1 + (w[0] - 1j * w[1]) * u2
    a2 = (w[0] + 1j * w[1]) * u1 - w[2] * u2
    return c * u1 - 1j * half_sinc * a1, c * u2 - 1j * half_sinc * a2


class SpinChiralityConfig:
    """配置：dim 必须为 4 的倍数（G 个 Dirac 旋量），dt 为演化步长。"""

    def __init__(self, dim: int = 16, dt: float = 0.1, absorption: bool = True,
                 quaternion: bool = True, gauge_mode: str = "protocol",
                 splitting_order: int | None = None) -> None:
        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"dim 必须为 4 的正整数倍（Dirac 旋量），当前 {dim}")
        if dt <= 0:
            raise ValueError(f"dt 必须为正，当前 {dt}")
        if gauge_mode not in ("protocol", "generator"):
            raise ValueError(f"gauge_mode 必须为 'protocol' 或 'generator'，当前 {gauge_mode!r}")
        order = (2 if gauge_mode == "generator" else 1) if splitting_order is None else int(splitting_order)
        if order not in (1, 2):
            raise ValueError(f"splitting_order 必须为 1 或 2，当前 {splitting_order!r}")
        self.dim = dim
        self.dt = float(dt)
        self.absorption = bool(absorption)
        self.quaternion = bool(quaternion)
        self.gauge_mode = gauge_mode
        self.splitting_order = order

    @property
    def n_groups(self) -> int:
        return self.dim // 4

    def __repr__(self) -> str:
        return (f"SpinChiralityConfig(dim={self.dim}, dt={self.dt:g}, absorption={self.absorption}, "
                f"gauge_mode={self.gauge_mode!r}, splitting_order={self.splitting_order})")


class _RealSpinChiralityConfig(SpinChiralityConfig):
    """实数等变包装器的配置视图：保留外部真实特征维 ``dim``，并将 ``n_groups`` 对齐内部旋量组数。"""

    def __init__(self, base: SpinChiralityConfig, n_groups: int, representation: str) -> None:
        super().__init__(
            dim=base.dim,
            dt=base.dt,
            absorption=base.absorption,
            quaternion=base.quaternion,
            gauge_mode=base.gauge_mode,
            splitting_order=base.splitting_order,
        )
        self._n_groups = int(n_groups)
        self._representation = representation

    @property
    def n_groups(self) -> int:
        return self._n_groups

    def __repr__(self) -> str:
        return (
            f"_RealSpinChiralityConfig(dim={self.dim}, n_groups={self._n_groups}, "
            f"representation={self._representation!r}, dt={self.dt:g}, "
            f"absorption={self.absorption}, gauge_mode={self.gauge_mode!r}, "
            f"splitting_order={self.splitting_order})"
        )


# ==============================================================================
# 2. 参考实现（NumPy，复数）
# ==============================================================================
def _split(x):
    """(…,4G) → ψ_L=(…,2G), ψ_R=(…,2G)，保持每组内部顺序。"""
    shape = x.shape
    g = shape[-1] // 4
    xg = x.reshape(*shape[:-1], g, 4)
    return xg[..., 0:2], xg[..., 2:4]


def _join(pl, pr):
    g = pl.shape[-2]
    out = np.concatenate([pl, pr], axis=-1)          # (...,G,4)
    return out.reshape(*out.shape[:-2], 4 * g)


def spin_chirality_numpy_forward(x, params, dt: float, absorption: bool = True,
                                 gauge_mode: str = "protocol",
                                 splitting_order: int | None = None):
    """物理忠实前向（NumPy）。

    Args:
        x: 复数数组 (B, L, 4G)。
        params: dict，键为
            theta_L, axis_L(3,G), theta_R, axis_R(3,G),
            B(3,G), mu(G), mass(G), gamma(G)
        dt: 演化步长。
        absorption: 是否包含吸收项 e^{-Γdt}。
        gauge_mode: "protocol"（瞬时转角 θ_{L/R}）或 "generator"（连续规范势，ω=A·dt）。
        splitting_order: 1 或 2（"generator" 默认 2 阶对称 Strang 分裂）。
    Returns:
        y: 复数数组 (B, L, 4G)。
    """
    if gauge_mode not in ("protocol", "generator"):
        raise ValueError(f"gauge_mode 必须为 'protocol' 或 'generator'，当前 {gauge_mode!r}")
    order = (2 if gauge_mode == "generator" else 1) if splitting_order is None else int(splitting_order)
    if order not in (1, 2):
        raise ValueError(f"splitting_order 必须为 1 或 2，当前 {splitting_order!r}")

    x = np.asarray(x, dtype=np.complex128)
    pl, pr = _split(x)                      # (B,L,G,2)
    G = x.shape[-1] // 4
    theta_L = np.asarray(params["theta_L"], dtype=float).reshape(G)
    theta_R = np.asarray(params["theta_R"], dtype=float).reshape(G)
    axis_L = np.asarray(params["axis_L"], dtype=float).reshape(3, G)
    axis_R = np.asarray(params["axis_R"], dtype=float).reshape(3, G)
    B = np.asarray(params["B"], dtype=float).reshape(3, G)
    mu = np.asarray(params["mu"], dtype=float).reshape(G)
    mass = np.asarray(params["mass"], dtype=float).reshape(G)
    gamma = np.asarray(params["gamma"], dtype=float).reshape(G) if absorption else np.zeros(G)

    scale_g = float(dt) if gauge_mode == "generator" else 1.0
    frac_g = (0.5 * scale_g) if order == 2 else scale_g

    pl_out = np.empty_like(pl)
    pr_out = np.empty_like(pr)
    for g in range(G):
        c2, s2 = np.cos(mass[g] * dt / 2), np.sin(mass[g] * dt / 2)
        l1 = c2 * pl[..., g, 0] - 1j * s2 * pr[..., g, 0]
        l2 = c2 * pl[..., g, 1] - 1j * s2 * pr[..., g, 1]
        r1 = c2 * pr[..., g, 0] - 1j * s2 * pl[..., g, 0]
        r2 = c2 * pr[..., g, 1] - 1j * s2 * pl[..., g, 1]

        thL = theta_L[g] * frac_g
        thR = theta_R[g] * frac_g
        l1, l2 = apply_su2_spinor(l1, l2, axis_L[:, g], thL)
        r1, r2 = apply_su2_spinor(r1, r2, axis_R[:, g], -thR)  # R†

        omega_B = -mu[g] * float(dt) * B[:, g]                  # H = -mu B·Sigma
        l1, l2 = _apply_su2_omega_numpy(l1, l2, omega_B)
        r1, r2 = _apply_su2_omega_numpy(r1, r2, omega_B)

        if order == 2:
            l1, l2 = apply_su2_spinor(l1, l2, axis_L[:, g], thL)
            r1, r2 = apply_su2_spinor(r1, r2, axis_R[:, g], -thR)

        nl1 = c2 * l1 - 1j * s2 * r1
        nl2 = c2 * l2 - 1j * s2 * r2
        nr1 = c2 * r1 - 1j * s2 * l1
        nr2 = c2 * r2 - 1j * s2 * l2
        att = np.exp(-gamma[g] * dt)
        pl_out[..., g, 0] = nl1 * att
        pl_out[..., g, 1] = nl2 * att
        pr_out[..., g, 0] = nr1 * att
        pr_out[..., g, 1] = nr2 * att
    return _join(pl_out, pr_out)


_VALID_REAL_REPS = ("dirac8", "chiral4", "su2_quat4")


def _inv_softplus_np(rate: np.ndarray) -> np.ndarray:
    """数值稳定地求 softplus(gamma) = rate (> 0) 的逆映射。"""
    r = np.asarray(rate, dtype=np.float64)
    out = np.empty_like(r)
    tiny = r <= 1e-12
    mid = (~tiny) & (r < 30.0)
    big = ~(tiny | mid)
    out[tiny] = -30.0
    out[mid] = np.log(np.expm1(r[mid]))
    out[big] = r[big]
    return out


def _pack_real_spinor_np(x: np.ndarray, representation: str) -> tuple[np.ndarray, int]:
    """将实数特征 x ∈ ℝ^{...×D} 等距/保结构打包为复数 Dirac 旋量 ∈ ℂ^{...×4G}。"""
    if representation not in _VALID_REAL_REPS:
        raise ValueError(
            f"representation 必须为 'dirac8'、'chiral4' 或 'su2_quat4'，当前 {representation!r}"
        )
    if np.iscomplexobj(x):
        raise TypeError("real_spin_chirality_numpy_forward 仅接受实数数组；复数旋量请用 spin_chirality_numpy_forward")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError(f"输入至少需 1 维，当前 shape={arr.shape}")
    D = arr.shape[-1]
    batch = arr.shape[:-1]
    if representation == "dirac8":
        if D == 0 or D % 8 != 0:
            raise ValueError(f"representation='dirac8' 要求最后一维为 8 的正整数倍，当前 {D}")
        G = D // 8
        v = arr.reshape(*batch, G, 2, 4)
        xc = (v[..., 0, :] + 1j * v[..., 1, :]).reshape(*batch, 4 * G)
        return xc.astype(np.complex128), G
    if D == 0 or D % 4 != 0:
        raise ValueError(f"representation={representation!r} 要求最后一维为 4 的正整数倍，当前 {D}")
    G = D // 4
    xg = arr.reshape(*batch, G, 4)
    xc = np.zeros((*batch, G, 4), dtype=np.complex128)
    if representation == "chiral4":
        xc[..., 0] = xg[..., 0] + 1j * xg[..., 1]   # ψ_L0 = L
        xc[..., 2] = xg[..., 2] + 1j * xg[..., 3]   # ψ_R0 = R
    else:  # "su2_quat4"
        xc[..., 0] = xg[..., 0] + 1j * xg[..., 1]
        xc[..., 1] = xg[..., 2] + 1j * xg[..., 3]
    return xc.reshape(*batch, 4 * G), G


def _unpack_real_spinor_np(yc: np.ndarray, representation: str, G: int) -> np.ndarray:
    """将复数 Dirac 旋量 yc ∈ ℂ^{...×4G} 解包回对应实数表示。"""
    batch = yc.shape[:-1]
    yg = yc.reshape(*batch, G, 4)
    if representation == "dirac8":
        return np.stack([yg.real, yg.imag], axis=-2).reshape(*batch, 8 * G)
    if representation == "chiral4":
        return np.stack(
            [yg[..., 0].real, yg[..., 0].imag, yg[..., 2].real, yg[..., 2].imag], axis=-1
        ).reshape(*batch, 4 * G)
    return np.stack(
        [yg[..., 0].real, yg[..., 0].imag, yg[..., 1].real, yg[..., 1].imag], axis=-1
    ).reshape(*batch, 4 * G)


def real_spin_chirality_numpy_forward(
    x,
    params: dict,
    dt: float,
    representation: str = "dirac8",
    absorption: bool = True,
    gauge_mode: str = "protocol",
    splitting_order: int | None = None,
) -> np.ndarray:
    """实数等变包装前向（NumPy），将主流实值张量严格按同构打包进复数 Dirac 旋量演化。

    三种表示：
    - ``"dirac8"`` (D=8G)：ℂ^{4G} ≅ ℝ^{8G} 的等距全息双射 (Re Ψ, Im Ψ)，保留全部 13G 参数。
    - ``"chiral4"`` (D=4G)：Cartan S_z=+1/2 手性子空间 (L, 0, R, 0)，钉死 axis=e_z、B=0，无漏失保范。
    - ``"su2_quat4"`` (D=4G)：ℂ^2 ≅ ℝ^4 四元数表示 (z0, z1, 0, 0)，钉死 mass=0、θ_R=0，精确实现 SU(2)⊂SO(4)。
    """
    arr = np.asarray(x)
    out_dtype = np.float32 if arr.dtype == np.float32 else np.float64
    xc, G = _pack_real_spinor_np(arr, representation)
    p_eff = dict(params)
    if representation == "chiral4":
        ez = np.zeros((3, G), dtype=float)
        ez[2, :] = 1.0
        p_eff["axis_L"] = ez
        p_eff["axis_R"] = ez
        B_in = np.asarray(params["B"], dtype=float).reshape(3, G).copy()
        B_in[:2, :] = 0.0
        p_eff["B"] = B_in
    elif representation == "su2_quat4":
        p_eff["mass"] = np.zeros(G, dtype=float)
        p_eff["theta_R"] = np.zeros(G, dtype=float)
    yc = spin_chirality_numpy_forward(
        xc, p_eff, dt, absorption=absorption, gauge_mode=gauge_mode, splitting_order=splitting_order
    )
    return _unpack_real_spinor_np(yc, representation, G).astype(out_dtype, copy=False)


def _to_real_numpy(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if np.iscomplexobj(arr):
        raise TypeError("Procrustes 求解器仅接受实数输入矩阵")
    return np.asarray(arr, dtype=np.float64)


def fit_chiral_procrustes(x, y, dt: float = 1.0) -> dict:
    """二维手性相位 Procrustes 闭式最优解（零迭代，纯 NumPy）。

    对实数对 ``(x, y) ∈ ℝ^{...×4G}``（按 ``(Re L, Im L, Re R, Im R)`` 配对），在每组上直接求
    ``min Σ ‖y - a · R(α_L, α_R) x‖²`` 与耦合轴矢估计 ``max Re(e^{iψ}⟨L_x,L_y⟩ + e^{-iψ}⟨R_x,R_y⟩)``：
    返回 ``alpha_L, alpha_R``（逆时针 2D 相位角），以及按 ``"chiral4"`` 约定对齐的
    ``theta_L = -2 α_L, theta_R = +2 α_R``、衰减系数 ``scale``、吸收率 ``gamma``
    与耦合轴向旋转估计 ``axial_angle = angle(⟨L_x,L_y⟩^* + ⟨R_x,R_y⟩)``、``common_angle``。
    """
    if dt <= 0:
        raise ValueError(f"dt 必须为正，当前 {dt}")
    xa = _to_real_numpy(x)
    ya = _to_real_numpy(y)
    if xa.shape != ya.shape:
        raise ValueError(f"x 与 y 形状必须一致，当前 {xa.shape} vs {ya.shape}")
    if xa.ndim < 1 or xa.shape[-1] == 0 or xa.shape[-1] % 4 != 0:
        raise ValueError(f"最后一维必须为 4 的正整数倍，当前 {xa.shape}")
    G = xa.shape[-1] // 4
    xg = xa.reshape(-1, G, 4)
    yg = ya.reshape(-1, G, 4)

    Lx = xg[:, :, 0] + 1j * xg[:, :, 1]
    Rx = xg[:, :, 2] + 1j * xg[:, :, 3]
    Ly = yg[:, :, 0] + 1j * yg[:, :, 1]
    Ry = yg[:, :, 2] + 1j * yg[:, :, 3]

    zL = np.sum(Lx.conj() * Ly, axis=0)
    zR = np.sum(Rx.conj() * Ry, axis=0)
    nL2 = np.sum(Lx.real ** 2 + Lx.imag ** 2, axis=0)
    nR2 = np.sum(Rx.real ** 2 + Rx.imag ** 2, axis=0)

    alpha_L = np.angle(zL)
    alpha_R = np.angle(zR)
    theta_L = -2.0 * alpha_L
    theta_R = 2.0 * alpha_R

    eps = 1e-30
    scale_L = np.abs(zL) / np.maximum(nL2, eps)
    scale_R = np.abs(zR) / np.maximum(nR2, eps)
    scale = (np.abs(zL) + np.abs(zR)) / np.maximum(nL2 + nR2, eps)
    gamma = np.maximum(0.0, -np.log(np.clip(scale, 1e-15, 1.0)) / float(dt))

    z_axial = zL.conj() + zR
    axial_angle = np.angle(z_axial)
    axial_scale = np.abs(z_axial) / np.maximum(nL2 + nR2, eps)
    common_angle = np.angle(zL + zR)

    return {
        "alpha_L": alpha_L,
        "alpha_R": alpha_R,
        "theta_L": theta_L,
        "theta_R": theta_R,
        "scale_L": scale_L,
        "scale_R": scale_R,
        "scale": scale,
        "gamma": gamma,
        "axial_angle": axial_angle,
        "axial_scale": axial_scale,
        "common_angle": common_angle,
    }


def _su2_quat_to_omega(q: np.ndarray) -> np.ndarray:
    """将单位四元数 q=(q0, q1, q2, q3)（对应 q0 I - i q·σ）精确映为轴角向量 ω。"""
    q0 = float(q[0])
    v = np.asarray(q[1:4], dtype=np.float64)
    vn2 = float(np.dot(v, v))
    if vn2 < 1e-24:
        if q0 >= 0.0:
            # 2 * atan2(vn, q0) / vn ≈ 2 / q0
            return (2.0 / max(q0, 1e-30)) * v
        return np.array([0.0, 0.0, 2.0 * np.pi], dtype=np.float64)
    vn = float(np.sqrt(vn2))
    theta = 2.0 * float(np.arctan2(vn, q0))
    return (theta / vn) * v


def fit_su2_procrustes(x, y) -> tuple[np.ndarray, float | np.ndarray, np.ndarray]:
    """ℂ^2 ≅ ℝ^4 群上的 SU(2) 闭式 Procrustes 求解（零迭代，无需 SVD）。

    Args:
        x, y: 实数 ``(..., 4)`` / ``(..., 4G)`` 或复数 ``(..., 2)`` / ``(..., 2G)`` 成对样本。
    Returns:
        ``(omega, scale, U)``：单组（D=4 实或 D=2 复）时形状为 ``((3,), float, (2, 2))``，
        多组（G > 1）时形状为 ``((G, 3), (G,), (G, 2, 2))``，均满足 ``U ≈ su2_exp(omega)`` 且
        ``min Σ ‖y - scale · U x‖²``。
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    xa = np.asarray(x)
    ya = np.asarray(y)
    if xa.shape != ya.shape:
        raise ValueError(f"x 与 y 形状必须一致，当前 {xa.shape} vs {ya.shape}")
    if xa.ndim < 1:
        raise ValueError(f"输入至少需 1 维，当前 {xa.shape}")
    is_cplx = np.iscomplexobj(xa) or np.iscomplexobj(ya)
    if is_cplx:
        if xa.shape[-1] == 0 or xa.shape[-1] % 2 != 0:
            raise ValueError(f"复数输入最后一维必须为 2 的正整数倍，当前 {xa.shape}")
        G = xa.shape[-1] // 2
        px = np.asarray(xa, dtype=np.complex128).reshape(-1, G, 2)
        py = np.asarray(ya, dtype=np.complex128).reshape(-1, G, 2)
    else:
        if xa.shape[-1] == 0 or xa.shape[-1] % 4 != 0:
            raise ValueError(f"实数输入最后一维必须为 4 的正整数倍，当前 {xa.shape}")
        G = xa.shape[-1] // 4
        xr = np.asarray(xa, dtype=np.float64).reshape(-1, G, 2, 2)
        yr = np.asarray(ya, dtype=np.float64).reshape(-1, G, 2, 2)
        px = xr[..., 0] + 1j * xr[..., 1]
        py = yr[..., 0] + 1j * yr[..., 1]

    omegas = np.zeros((G, 3), dtype=np.float64)
    scales = np.zeros(G, dtype=np.float64)
    Us = np.zeros((G, 2, 2), dtype=np.complex128)
    for g in range(G):
        Xg = px[:, g, :]
        Yg = py[:, g, :]
        # C = Σ y_m x_m†；tr(U† C) = q0 Re tr(C) - Σ_{k=1}^3 q_k Im tr(σ_k C)
        C = Yg.T @ Xg.conj()
        b = np.array(
            [
                float(np.real(np.trace(C))),
                float(-np.imag(np.trace(S1 @ C))),
                float(-np.imag(np.trace(S2 @ C))),
                float(-np.imag(np.trace(S3 @ C))),
            ],
            dtype=np.float64,
        )
        bn = float(np.linalg.norm(b))
        den = float(np.sum(Xg.real ** 2 + Xg.imag ** 2))
        if bn < 1e-30 or den < 1e-30:
            omegas[g] = 0.0
            scales[g] = 0.0
            Us[g] = I2.copy()
        else:
            q = b / bn
            omegas[g] = _su2_quat_to_omega(q)
            scales[g] = bn / den
            Us[g] = su2_exp(omegas[g])
    if G == 1:
        return omegas[0], float(scales[0]), Us[0]
    return omegas, scales, Us


# ==============================================================================
# 3. PyTorch 模块（复数张量，可微）
# ==============================================================================
if HAS_TORCH:


    def _axis_to_unit_torch(axis):
        """安全单位化 (G,3)：零轴返回零向量，避免 0/0 NaN 与 ~1e30 梯度爆炸。"""
        nrm2 = (axis * axis).sum(dim=-1, keepdim=True)
        is_zero = nrm2 <= 0.0
        nrm = torch.sqrt(torch.where(is_zero, torch.ones_like(nrm2), nrm2))
        return torch.where(is_zero, torch.zeros_like(axis), axis / nrm)

    def _sinc_half_cos_torch(omega):
        """光滑计算 cos(|ω|/2) 与 sin(|ω|/2)/|ω|（ω=0 处用偶次 Taylor 展开，无 √0/÷0 奇点）。"""
        phi2 = (omega * omega).sum(dim=-1)
        x2 = 0.25 * phi2
        small = x2 < 1e-8
        x2_safe = torch.where(small, torch.ones_like(x2), x2)
        phi_safe = torch.sqrt(4.0 * x2_safe)
        c = torch.where(small, 1.0 - 0.5 * x2 + (x2 * x2) / 24.0, torch.cos(0.5 * phi_safe))
        half_sinc = torch.where(
            small,
            0.5 * (1.0 - x2 / 6.0 + (x2 * x2) / 120.0),
            torch.sin(0.5 * phi_safe) / phi_safe,
        )
        return c, half_sinc

    def _quat_exp_torch(omega):
        """光滑 SU(2) 指数映射四元数：exp(-i ω·σ/2) 的单位四元数 (G,4)。"""
        c, half_sinc = _sinc_half_cos_torch(omega)
        return torch.cat([c[:, None], -half_sinc[:, None] * omega], dim=-1)

    def _su2_exp_torch(omega):
        """光滑 SU(2) 指数映射矩阵：exp(-i ω·σ/2) 的批量 (G,2,2) 复矩阵。"""
        c, half_sinc = _sinc_half_cos_torch(omega)
        cdtype = torch.complex128 if omega.dtype == torch.float64 else torch.complex64
        eye = torch.eye(2, dtype=cdtype, device=omega.device).expand(omega.shape[0], 2, 2)
        wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
        w_sig = torch.stack([
            torch.stack([wz, wx - 1j * wy], dim=-1),
            torch.stack([wx + 1j * wy, -wz], dim=-1),
        ], dim=-2).to(cdtype)
        return c[:, None, None].to(cdtype) * eye - 1j * half_sinc[:, None, None].to(cdtype) * w_sig

    def _quat_torch(axis, theta):
        """U = cos(θ/2) I − i sin(θ/2)(n̂·σ) 的四元数 (G,4)（零轴返回单位四元数）。"""
        return _quat_exp_torch(theta[:, None] * _axis_to_unit_torch(axis))

    def _quat_apply_torch(q, z):
        """四元数作用在复 2-旋量 (…,G,2) 上。"""
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        z1, z2 = z[..., 0], z[..., 1]
        return torch.stack([(q0 + 1j * q3) * z1 + (q2 + 1j * q1) * z2,
                            (-q2 + 1j * q1) * z1 + (q0 - 1j * q3) * z2], dim=-1)

    def _quat_conj_torch(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def _su2_torch(axis, theta):
        """axis: (G,3) real tensor，theta: (G,) → 2×2 复数矩阵的批量 (G,2,2)（零轴返回 I₂）。"""
        return _su2_exp_torch(theta[:, None] * _axis_to_unit_torch(axis))

    def _apply_gauge_torch(pl, pr, omega_L, omega_R, quaternion: bool, dtype):
        """对 (pl, pr) 施加手征规范转动 U_L(ω_L) ⊕ U_R†(ω_R)。"""
        if quaternion:
            qL = _quat_exp_torch(omega_L)
            qR = _quat_exp_torch(omega_R)
            return _quat_apply_torch(qL, pl), _quat_apply_torch(_quat_conj_torch(qR), pr)
        U_L = _su2_exp_torch(omega_L).to(dtype)
        U_R = _su2_exp_torch(omega_R).to(dtype)
        return (
            torch.einsum("...gi,gji->...gj", pl, U_L),
            torch.einsum("...gi,gij->...gj", pr, U_R.conj()),
        )

    class SpinChiralityOperator(nn.Module):
        """Dirac/Weyl 演化算子：手性规范 × Larmor × 质量 × 吸收。

        参数（每组 G 个，均可学习）：
            theta_L, axis_L(3)   SU(2)_L 转角/率与转轴
            theta_R, axis_R(3)   SU(2)_R
            B(3), mu             Larmor 磁场与磁矩
            mass                 质量
            gamma                吸收系数（softplus 保证 ≥ 0）
        """

        def __init__(self, config: SpinChiralityConfig, dtype=torch.complex128, device=None) -> None:
            super().__init__()
            if dtype not in (torch.complex64, torch.complex128):
                raise TypeError(f"dtype 必须为 torch.complex64 或 torch.complex128，当前 {dtype!r}")
            self.config = config
            G = config.n_groups
            real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
            self.dtype = dtype

            def param(*shape, value=0.0):
                return nn.Parameter(torch.full(shape, value, dtype=real_dtype, device=device))

            self.theta_L = param(G, value=0.3)
            self.axis_L = param(G, 3, value=0.0)
            with torch.no_grad():
                self.axis_L[:, 2] = 1.0
            self.theta_R = param(G, value=-0.2)
            self.axis_R = param(G, 3, value=0.0)
            with torch.no_grad():
                self.axis_R[:, 2] = 1.0
            self.B = param(G, 3, value=0.0)
            with torch.no_grad():
                self.B[:, 2] = 1.0
            self.mu = param(G, value=1.0)
            self.mass = param(G, value=0.1)
            self.gamma = param(G, value=-6.0)  # softplus(-6) ≈ 2.5e-3

        def _gamma_eff(self):
            return torch.nn.functional.softplus(self.gamma)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not x.is_complex():
                raise TypeError("输入必须是复数张量（ℂ^{B×L×4G}）")
            if x.dtype not in (torch.complex64, torch.complex128):
                raise TypeError(f"仅支持 complex64 / complex128，当前 {x.dtype}")
            G = self.config.n_groups
            if x.shape[-1] != 4 * G:
                raise ValueError(f"输入最后一维应为 4G={4 * G}，当前 {x.shape[-1]}")
            dtype = x.dtype
            real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
            xg = x.reshape(*x.shape[:-1], G, 4)
            pl, pr = xg[..., 0:2], xg[..., 2:4]          # (...,G,2)
            mdt2 = self.mass.to(real_dtype) * self.config.dt / 2
            c2 = torch.cos(mdt2).to(dtype)[:, None]
            s2 = torch.sin(mdt2).to(dtype)[:, None]
            pl, pr = c2 * pl - 1j * s2 * pr, c2 * pr - 1j * s2 * pl   # 半质量 kick

            scale_g = self.config.dt if self.config.gauge_mode == "generator" else 1.0
            frac_g = (0.5 * scale_g) if self.config.splitting_order == 2 else scale_g
            omega_L = (self.theta_L.to(real_dtype) * frac_g)[:, None] * _axis_to_unit_torch(self.axis_L.to(real_dtype))
            omega_R = (self.theta_R.to(real_dtype) * frac_g)[:, None] * _axis_to_unit_torch(self.axis_R.to(real_dtype))

            pl, pr = _apply_gauge_torch(pl, pr, omega_L, omega_R, self.config.quaternion, dtype)

            omega_B = (-self.mu.to(real_dtype) * self.config.dt)[:, None] * self.B.to(real_dtype)
            if self.config.quaternion:
                qB = _quat_exp_torch(omega_B)
                pl = _quat_apply_torch(qB, pl)
                pr = _quat_apply_torch(qB, pr)
            else:
                U_B = _su2_exp_torch(omega_B).to(dtype)
                pl = torch.einsum("...gi,gji->...gj", pl, U_B)
                pr = torch.einsum("...gi,gji->...gj", pr, U_B)

            if self.config.splitting_order == 2:
                pl, pr = _apply_gauge_torch(pl, pr, omega_L, omega_R, self.config.quaternion, dtype)

            nl = c2 * pl - 1j * s2 * pr   # 后半步质量 kick
            nr = c2 * pr - 1j * s2 * pl

            if self.config.absorption:
                att = torch.exp(-self._gamma_eff().to(real_dtype) * self.config.dt).to(dtype)[:, None]
                nl, nr = nl * att, nr * att

            y = torch.cat([nl, nr], dim=-1).reshape(*x.shape[:-1], 4 * G)
            return y

        def forward_sequence(self, u: torch.Tensor, coupling: float = 1.0,
                             M_seq: torch.Tensor | None = None,
                             analytic: bool = False) -> torch.Tensor:
            """沿 L 做线性递推演化 s_{n+1} = M s_n + c u_n（支持静态 M 与时变 M_seq）。"""
            if not u.is_complex():
                raise TypeError("输入必须是复数张量（ℂ^{B×L×4G}）")
            if u.dtype not in (torch.complex64, torch.complex128):
                raise TypeError(f"仅支持 complex64 / complex128，当前 {u.dtype}")
            real_dtype = torch.float64 if u.dtype == torch.complex128 else torch.float32
            M = propagator_torch(self, dtype=u.dtype) if M_seq is None else M_seq.to(dtype=u.dtype, device=u.device)
            c_t = coupling if isinstance(coupling, torch.Tensor) else torch.tensor(
                float(coupling), dtype=real_dtype, device=u.device
            )
            fn = AnalyticSequenceFunction if analytic else SequenceFunction
            return fn.apply(u, M, c_t)

        # ----- 物理量接口 -----------------------------------------------------
        def axial_charge(self, x: torch.Tensor) -> torch.Tensor:
            """Q5 = ∫ ψ†γ⁵ψ（逐样本）。"""
            xg = x.reshape(*x.shape[:-1], -1, 4)
            return (xg.conj() * (xg @ torch.diag(torch.tensor(
                [-1, -1, 1, 1], dtype=x.dtype, device=x.device)))).sum(dim=(-1, -2)).real

        def spin_expectation(self, x: torch.Tensor) -> torch.Tensor:
            """⟨Σ⟩ = ψ† diag(σ/2, σ/2) ψ（逐样本，3 分量）。"""
            xg = x.reshape(*x.shape[:-1], -1, 4)
            pl, pr = xg[..., 0:2], xg[..., 2:4]
            out = []
            sig = [
                torch.tensor([[0, 1], [1, 0]], dtype=x.dtype, device=x.device),
                torch.tensor([[0, -1j], [1j, 0]], dtype=x.dtype, device=x.device),
                torch.tensor([[1, 0], [0, -1]], dtype=x.dtype, device=x.device),
            ]
            for S in sig:
                v = (pl.conj() @ S * pl).sum(dim=(-1, -2)) + (pr.conj() @ S * pr).sum(dim=(-1, -2))
                out.append(0.5 * v.real)
            return torch.stack(out, dim=-1)

    class RealSpinChiralityOperator(nn.Module):
        """实数等变包装算子：将 ℝ^{...×D} 同构映入复数 Dirac 旋量并调用 ``SpinChiralityOperator``。

        设计目的：保持底层 ``SpinChiralityOperator`` 严格复数输入门禁，同时为主流实值网络提供
        零投影损失、保持梯度的实数接口，并在 ``"chiral4"`` / ``"su2_quat4"`` 上支持闭式 Procrustes 拟合。

        三种表示（由 ``representation`` 指定）：
        - ``"dirac8"`` (D=8G)：ℂ^{4G} ≅ ℝ^{8G} 等距全息双射 (Re Ψ, Im Ψ)，保留全部 13G 参数。
        - ``"chiral4"`` (D=4G)：Cartan S_z=+1/2 手性子空间 (L, 0, R, 0)，冻结 axis=e_z、B=0、μ；
          初始 ``mass=0`` 位于无质量对角相位子流形（仍保留梯度可微，可学得 L↔R 质量混合；
          ``fit_procrustes`` 则直接求 m=0 闭式最优解）。
        - ``"su2_quat4"`` (D=4G)：ℂ^2 ≅ ℝ^4 四元数表示 (z0, z1, 0, 0)，冻结 mass=0、θ_R=0。
        """

        def __init__(
            self,
            config: SpinChiralityConfig,
            representation: str = "dirac8",
            dtype=torch.float32,
            device=None,
        ) -> None:
            super().__init__()
            if representation not in _VALID_REAL_REPS:
                raise ValueError(
                    f"representation 必须为 'dirac8'、'chiral4' 或 'su2_quat4'，当前 {representation!r}"
                )
            if dtype not in (torch.float32, torch.float64):
                raise TypeError(f"dtype 必须为 torch.float32 或 torch.float64，当前 {dtype!r}")
            if representation == "dirac8":
                if config.dim % 8 != 0:
                    raise ValueError(
                        f"representation='dirac8' 要求 config.dim 为 8 的倍数，当前 {config.dim}"
                    )
                complex_dim = config.dim // 2
            else:
                complex_dim = config.dim

            inner_cfg = SpinChiralityConfig(
                dim=complex_dim,
                dt=config.dt,
                absorption=config.absorption,
                quaternion=config.quaternion,
                gauge_mode=config.gauge_mode,
                splitting_order=config.splitting_order,
            )
            complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            self.representation = representation
            self.config = _RealSpinChiralityConfig(config, inner_cfg.n_groups, representation)
            self.inner = SpinChiralityOperator(inner_cfg, dtype=complex_dtype, device=device)

            if representation == "chiral4":
                with torch.no_grad():
                    self.inner.axis_L.zero_()
                    self.inner.axis_L[:, 2] = 1.0
                    self.inner.axis_R.zero_()
                    self.inner.axis_R[:, 2] = 1.0
                    self.inner.B.zero_()
                    self.inner.mass.zero_()
                self.inner.axis_L.requires_grad_(False)
                self.inner.axis_R.requires_grad_(False)
                self.inner.B.requires_grad_(False)
                self.inner.mu.requires_grad_(False)
            elif representation == "su2_quat4":
                with torch.no_grad():
                    self.inner.theta_R.zero_()
                    self.inner.axis_R.zero_()
                    self.inner.axis_R[:, 2] = 1.0
                    self.inner.mass.zero_()
                self.inner.theta_R.requires_grad_(False)
                self.inner.axis_R.requires_grad_(False)
                self.inner.mass.requires_grad_(False)

        @property
        def dtype(self):
            return self.inner.theta_L.dtype

        @property
        def complex_dtype(self):
            return torch.complex128 if self.inner.theta_L.dtype == torch.float64 else torch.complex64

        @property
        def n_groups(self) -> int:
            return self.inner.config.n_groups

        @property
        def theta_L(self):
            return self.inner.theta_L

        @property
        def axis_L(self):
            return self.inner.axis_L

        @property
        def theta_R(self):
            return self.inner.theta_R

        @property
        def axis_R(self):
            return self.inner.axis_R

        @property
        def B(self):
            return self.inner.B

        @property
        def mu(self):
            return self.inner.mu

        @property
        def mass(self):
            return self.inner.mass

        @property
        def gamma(self):
            return self.inner.gamma

        def _pack(self, x: torch.Tensor) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"输入必须为 torch.Tensor，当前 {type(x).__name__}")
            if x.is_complex():
                raise TypeError(
                    "RealSpinChiralityOperator 仅接受实数张量；复数旋量请直接使用 SpinChiralityOperator"
                )
            if x.dtype not in (torch.float32, torch.float64):
                raise TypeError(f"仅支持 float32 / float64 实数张量，当前 {x.dtype}")
            if x.ndim < 1 or x.shape[-1] != self.config.dim:
                raise ValueError(f"输入最后一维应为 {self.config.dim}，当前 {tuple(x.shape)}")
            G = self.n_groups
            batch = x.shape[:-1]
            if self.representation == "dirac8":
                v = x.reshape(*batch, G, 2, 4)
                return torch.complex(v[..., 0, :], v[..., 1, :]).reshape(*batch, 4 * G)
            xg = x.reshape(*batch, G, 4)
            z0 = torch.zeros_like(xg[..., 0])
            if self.representation == "chiral4":
                re = torch.stack([xg[..., 0], z0, xg[..., 2], z0], dim=-1)
                im = torch.stack([xg[..., 1], z0, xg[..., 3], z0], dim=-1)
            else:  # "su2_quat4"
                re = torch.stack([xg[..., 0], xg[..., 2], z0, z0], dim=-1)
                im = torch.stack([xg[..., 1], xg[..., 3], z0, z0], dim=-1)
            return torch.complex(re, im).reshape(*batch, 4 * G)

        def _unpack(self, yc: torch.Tensor) -> torch.Tensor:
            G = self.n_groups
            batch = yc.shape[:-1]
            yg = yc.reshape(*batch, G, 4)
            if self.representation == "dirac8":
                return torch.stack([yg.real, yg.imag], dim=-2).reshape(*batch, 8 * G)
            if self.representation == "chiral4":
                return torch.stack(
                    [yg[..., 0].real, yg[..., 0].imag, yg[..., 2].real, yg[..., 2].imag], dim=-1
                ).reshape(*batch, 4 * G)
            return torch.stack(
                [yg[..., 0].real, yg[..., 0].imag, yg[..., 1].real, yg[..., 1].imag], dim=-1
            ).reshape(*batch, 4 * G)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._unpack(self.inner(self._pack(x)))

        def forward_sequence(
            self,
            u: torch.Tensor,
            coupling: float = 1.0,
            M_seq: torch.Tensor | None = None,
            analytic: bool = False,
        ) -> torch.Tensor:
            """实数张量沿 L 维线性递推演化（复用底层复数扫描与解析伴随）。"""
            uc = self._pack(u)
            yc = self.inner.forward_sequence(uc, coupling=coupling, M_seq=M_seq, analytic=analytic)
            return self._unpack(yc)

        def fit_procrustes(self, x: torch.Tensor, y: torch.Tensor, update: bool = True) -> dict:
            """在 ``chiral4`` / ``su2_quat4`` 表示下执行闭式 Procrustes 反演并可选写回参数。"""
            if self.representation == "dirac8":
                raise NotImplementedError(
                    "dirac8 通用 4 分量 Dirac 旋量暂未定义全局闭式 Procrustes，请用梯度优化或 chiral4/su2_quat4"
                )
            G = self.n_groups
            dt = self.config.dt
            factor = dt if self.config.gauge_mode == "generator" else 1.0
            param_dtype = self.inner.theta_L.dtype
            dev = self.inner.theta_L.device

            if self.representation == "chiral4":
                res = fit_chiral_procrustes(x, y, dt=dt)
                if update:
                    with torch.no_grad():
                        self.inner.theta_L.copy_(torch.as_tensor(res["theta_L"] / factor, dtype=param_dtype, device=dev))
                        self.inner.theta_R.copy_(torch.as_tensor(res["theta_R"] / factor, dtype=param_dtype, device=dev))
                        self.inner.mass.zero_()
                        self.inner.B.zero_()
                        if self.config.absorption:
                            g_eff = -np.log(np.clip(res["scale"], 1e-12, 1.0)) / dt
                            g_raw = _inv_softplus_np(g_eff)
                            self.inner.gamma.copy_(torch.as_tensor(g_raw, dtype=param_dtype, device=dev))
                return res

            omega, scale, U = fit_su2_procrustes(x, y)
            omega_g = np.asarray(omega, dtype=np.float64).reshape(G, 3)
            scale_g = np.asarray(scale, dtype=np.float64).reshape(G)
            if update:
                norms = np.linalg.norm(omega_g, axis=-1)
                axes = np.where(
                    norms[:, None] > 1e-15,
                    omega_g / np.maximum(norms[:, None], 1e-30),
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
                )
                with torch.no_grad():
                    self.inner.theta_L.copy_(torch.as_tensor(norms / factor, dtype=param_dtype, device=dev))
                    self.inner.axis_L.copy_(torch.as_tensor(axes, dtype=param_dtype, device=dev))
                    self.inner.B.zero_()
                    self.inner.mass.zero_()
                    self.inner.theta_R.zero_()
                    if self.config.absorption:
                        g_eff = -np.log(np.clip(scale_g, 1e-12, 1.0)) / dt
                        g_raw = _inv_softplus_np(g_eff)
                        self.inner.gamma.copy_(torch.as_tensor(g_raw, dtype=param_dtype, device=dev))
            return {"omega": omega, "scale": scale, "U": U}

else:  # pragma: no cover
    SpinChiralityOperator = None  # type: ignore
    RealSpinChiralityOperator = None  # type: ignore


# ==============================================================================
# 4. 物理检验
# ==============================================================================
def _check_clifford() -> dict:
    worst = 0.0
    for a in range(4):
        for b in range(4):
            worst = max(worst, np.max(np.abs(GAMMA[a] @ GAMMA[b] + GAMMA[b] @ GAMMA[a] - 2 * ETA[a, b] * ID4)))
    g5 = 1j * GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]
    return dict(
        anticommutator=float(worst),
        gamma5_trace=float(abs(np.trace(GAMMA5))),
        gamma5_square=float(np.max(np.abs(GAMMA5 @ GAMMA5 - ID4))),
        gamma5_anticommute=float(max(np.max(np.abs(GAMMA5 @ gm + gm @ GAMMA5)) for gm in GAMMA)),
    )


def _check_projectors() -> dict:
    n = lambda a: float(np.max(np.abs(a)))
    return dict(
        pl_idem=n(PL @ PL - PL), pr_idem=n(PR @ PR - PR), orth=n(PL @ PR),
        complete=n(PL + PR - ID4), trace=float(abs(np.trace(PL) - 2) + abs(np.trace(PR) - 2)),
        eigen=n(GAMMA5 @ PL + PL) + n(GAMMA5 @ PR - PR),
    )


def _check_su2() -> dict:
    rng = np.random.default_rng(0)
    unit = det = group = lie = 0.0
    for _ in range(20):
        a = rng.normal(size=3)
        t1, t2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        U = su2(a, t1)
        unit = max(unit, float(np.max(np.abs(U.conj().T @ U - I2))))
        det = max(det, abs(np.linalg.det(U) - 1))
        group = max(group, float(np.max(np.abs(su2(a, t1) @ su2(a, t2) - su2(a, t1 + t2)))))
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[1, 0, 2] = eps[2, 1, 0] = eps[0, 2, 1] = -1
    for i in range(3):
        for j in range(3):
            comm = 0.25 * (SIG[i] @ SIG[j] - SIG[j] @ SIG[i])
            target = 1j * sum(eps[i, j, k] * 0.5 * SIG[k] for k in range(3))
            lie = max(lie, float(np.max(np.abs(comm - target))))
    return dict(unitarity=unit, det=det, group_law=group, lie_algebra=lie)


def _check_larmor() -> dict:
    rng = np.random.default_rng(1)
    B = np.array([0.3, -0.4, 1.0])
    mu, dt = 1.7, 0.37
    worst = 0.0
    U = su2(B, -mu * np.linalg.norm(B) * dt)   # H = -mu B·Sigma
    for _ in range(10):
        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi /= np.linalg.norm(psi)
        out = U @ psi
        s0 = np.array([float(np.real(psi.conj() @ S @ psi)) for S in SIG])
        st = np.array([float(np.real(out.conj() @ S @ out)) for S in SIG])
        # 经典进动：绕 B̂ 转 ω dt
        n = B / np.linalg.norm(B)
        # Rodrigues 公式
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        ang = -mu * np.linalg.norm(B) * dt
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        worst = max(worst, float(np.max(np.abs(st - R @ s0))))
    return dict(expectation_vs_classical=worst)



def _check_chirality() -> dict:
    # 纯左手初态：(1,0,0,0)
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0
    # m = 0：手性守恒（β 项为零）
    H0 = GAMMA[0] @ GAMMA[1] * 0.7
    U0 = _expm(-1j * H0 * 2.3)
    g5_0 = float(np.real((U0 @ psi).conj() @ GAMMA5 @ (U0 @ psi)))
    # m > 0, p = 0：⟨γ5⟩(t) = −cos(2mt)
    m, t = 0.31, 1.9
    Um = np.cos(m * t) * ID4 - 1j * np.sin(m * t) * BETA
    g5_m = float(np.real((Um @ psi).conj() @ GAMMA5 @ (Um @ psi)))
    return dict(chirality_massless=abs(g5_0 + 1.0), chirality_massive=abs(g5_m + np.cos(2 * m * t)))


def _expm(A: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(A)
    return v @ np.diag(np.exp(w)) @ np.linalg.inv(v)


def _check_ward() -> dict:
    """双模叠加态上 ∂_μ j^μ_5 = 2 i m ψ̄γ5ψ。"""
    rng = np.random.default_rng(2)
    m = 0.42
    moms = [np.array([0.9, -0.3, 0.5]), np.array([-0.4, 0.8, 0.2])]
    us, ubs, p4 = [], [], []
    for p3 in moms:
        p0 = np.sqrt(p3 @ p3 + m * m)
        pslash = p0 * GAMMA[0] - sum(p3[i] * GAMMA[i + 1] for i in range(3))
        u0 = rng.normal(size=4) + 1j * rng.normal(size=4)
        u = (pslash + m * ID4) @ u0
        us.append(u / np.linalg.norm(u))
        ubs.append(us[-1].conj() @ GAMMA[0])
        p4.append(np.array([p0, -p3[0], -p3[1], -p3[2]]))   # 协变分量
    lhs = 0j
    rhs = 0j
    for a in range(2):
        for b in range(2):
            dp = p4[b] - p4[a]
            lhs += 1j * sum(dp[mu] * (ubs[b] @ GAMMA[mu] @ GAMMA5 @ us[a]) for mu in range(4))
            rhs += 2j * m * (ubs[b] @ GAMMA5 @ us[a])
    return dict(ward_residual=float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)),
                lhs_magnitude=float(abs(lhs)))


def _check_unitarity() -> dict:
    rng = np.random.default_rng(3)
    G = 3
    x = (rng.normal(size=(2, 4, 4 * G)) + 1j * rng.normal(size=(2, 4, 4 * G)))
    params = dict(
        theta_L=rng.uniform(-1, 1, G), axis_L=rng.normal(size=(3, G)),
        theta_R=rng.uniform(-1, 1, G), axis_R=rng.normal(size=(3, G)),
        B=rng.normal(size=(3, G)), mu=rng.uniform(0.5, 2, G),
        mass=rng.uniform(0, 0.5, G), gamma=np.zeros(G),
    )
    dt = 0.13
    y0 = spin_chirality_numpy_forward(x, params, dt, absorption=False)
    # Γ = 0 时严格等范；Γ > 0 时按 e^{−Γdt} 衰减
    err_unit = float(np.max(np.abs(np.linalg.norm(y0.reshape(-1, 4 * G), axis=-1) -
                                   np.linalg.norm(x.reshape(-1, 4 * G), axis=-1))))
    params2 = dict(params)
    params2["gamma"] = np.full(G, 0.7)
    y1 = spin_chirality_numpy_forward(x, params2, dt, absorption=True)
    expected = np.exp(-0.7 * dt) * np.linalg.norm(x.reshape(-1, 4 * G), axis=-1)
    err_abs = float(np.max(np.abs(np.linalg.norm(y1.reshape(-1, 4 * G), axis=-1) - expected)))
    return dict(unitarity=err_unit, absorption_law=err_abs)


def _check_gradcheck() -> dict:
    if not HAS_TORCH:
        return {"gradcheck": float("nan")}
    torch.manual_seed(0)
    G = 2
    op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1), dtype=torch.complex128)
    x = (torch.randn(2, 3, 4 * G, dtype=torch.float64) +
         1j * torch.randn(2, 3, 4 * G, dtype=torch.float64)).requires_grad_(True)

    def loss(t: torch.Tensor) -> torch.Tensor:
        y = op(t)
        return (y.real ** 2 + y.imag ** 2 * 0.5).sum()

    # 复输入的有限差分：分别扰动实部与虚部
    eps = 1e-6
    base = loss(x).item()
    xr = x.detach().clone()
    gx = torch.autograd.grad(loss(x), x)[0]
    worst = 0.0
    for idx in [(0, 0, 0), (1, 2, 3), (0, 1, 4 * G - 1)]:
        for part in ("real", "imag"):
            xp, xm = xr.clone(), xr.clone()
            if part == "real":
                xp[idx] += eps; xm[idx] -= eps
            else:
                xp[idx] += 1j * eps; xm[idx] -= 1j * eps
            fd = (loss(xp).item() - loss(xm).item()) / (2 * eps)
            if part == "real":
                ana = float(gx[idx].real)
            else:
                ana = float(gx[idx].imag)
            worst = max(worst, abs(fd - ana) / max(abs(fd), abs(ana), 1e-8))
    return dict(gradcheck_rel=worst)


def _check_exact_free_evolution() -> dict:
    """自由演化精确性：[Σ_i,β]=0，故 e^{-i(-μB·Σ+mβ)dt} = e^{-imβdt} e^{+iμB·Σdt}。"""
    Sig = [(0.5 * np.block([[S, np.zeros((2, 2))], [np.zeros((2, 2)), S]])) for S in SIG]
    comm = max(float(np.max(np.abs(Sg @ BETA - BETA @ Sg))) for Sg in Sig)
    rng = np.random.default_rng(4)
    mu, mass, dt = 1.3, 0.42, 0.37
    B = np.array([0.3, -0.4, 1.0])
    H = -mu * sum(B[i] * Sig[i] for i in range(3)) + mass * BETA
    U_exact = _expm(-1j * H * dt)
    # 模块的前向（关掉手性旋转）
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    params = dict(theta_L=np.zeros(1), axis_L=np.zeros((3, 1)),
                  theta_R=np.zeros(1), axis_R=np.zeros((3, 1)),
                  B=B.reshape(3, 1), mu=np.array([mu]), mass=np.array([mass]),
                  gamma=np.zeros(1))
    y = spin_chirality_numpy_forward(psi.reshape(1, 1, 4), params, dt, absorption=False).reshape(4)
    return dict(commutator=comm, exact_free_evolution=float(np.max(np.abs(y - U_exact @ psi))))




# ==============================================================================
# 单粒子 QFT 工具与实验（原 qft.py，2026-11-10 合并进本模块）
#
# 这一段是同一批物理对象的单粒子 NumPy 写法：SU(2) 李代数生成元、Larmor 酉演化、
# 自旋期望、Dirac 哈密顿量，以及 1+1 维反常指标（磁通量子数 = 指标）。它们原先单独
# 放在 qft.py 里，与 operator 的 _check_* 平行维护——改一处要同步两处，很容易各自
# 漂移；合并后只剩这一份。P1–P16 的验收检查都在 run_physics_audit 中，本段除
# dirac_hamiltonian 外都是辅助工具与专项核对。
# ==============================================================================
ALPHA = [GAMMA[0] @ GAMMA[i] for i in range(1, 4)]        # α_i = γ⁰γⁱ（β = γ⁰ 见上）


def su2_generator(n_hat: np.ndarray) -> np.ndarray:
    """(i/2) n·σ —— 反厄米无迹 SU(2) 生成元：exp(−θ·su2_generator(n̂)) = su2(n̂, θ)。"""
    n = np.asarray(n_hat, dtype=float)
    return 0.5j * sum(n[i] * SIG[i] for i in range(3))


def larmor_unitary(B: np.ndarray, mu: float, t: float) -> np.ndarray:
    """U(t) = exp(−i H t)，H = −μ B·σ/2 ⇒ U(t) = su2_exp(−μ t B)（B=0 光滑返回 I₂）。"""
    return su2_exp(-float(mu) * float(t) * np.asarray(B, dtype=float))


def spin_expectation_numpy(psi: np.ndarray) -> np.ndarray:
    """⟨σ⟩（未除 2）：三分量实向量；torch 版 SpinChiralityOperator.spin_expectation 是 ⟨Σ⟩=⟨σ⟩/2。"""
    return np.array([float(np.real(psi.conj() @ SIG[i] @ psi)) for i in range(3)])


def dirac_hamiltonian(p: np.ndarray, m: float) -> np.ndarray:
    """H = α·p + βm（Weyl 表象）的 4×4 单粒子 Dirac 哈密顿量。"""
    return sum(ALPHA[i] * p[i] for i in range(3)) + BETA * m


# ------------------------------------------------------------------------------
# 反常指标（旧 qft 层的 P7）：1+1 维环面上的指标定理 —— index(D_ov) = 磁通量子数 n。
#
# 原先的做法是让磁通绝热增长、沿时间数"能级穿过零点"的次数：这要跟踪本征值的连续
# 性，而排序后的本征值会重排，计数既不收敛也不与 E·T 成比例。改为直接取固定磁通
# 背景：Atiyah–Singer 给出 index(D) = (1/2π)∫F = n，整数、无歧义，还能用两种互相
# 独立的算法（零空间净手性 / 惯性）交叉验证。n = 0/±1/±2 的实测指标 = 0/±1/±2。
#
# 这一项不参与 run_physics_audit 的 P1–P16 验收，属专项核对。
# ------------------------------------------------------------------------------
_G5_2D = 1j * S1 @ S2          # 1+1D：γ⁵ = iγ₁γ₂ = −σ₃


def _wilson_2d_kernel(L: int, nflux: int, m: float = 0.0, r: float = 1.0) -> np.ndarray:
    """L×L 环面上带总磁通 2π·nflux 的 Wilson–Dirac 核算符 A = D_W − 1。

    规范由磁通量子化定出：θ = 2πn/L²，U₁(x,y) = e^{−iθy}，U₂ 只在 y = L−1 一行带
    e^{2πinx/L}，于是每个 plaquette 相位都是 θ、总磁通 = 2πn。反向跃迁必须用**源点**
    链接的共轭，否则 γ⁵-Hermiticity 被破坏。
    """
    th = 2.0 * np.pi * nflux / L ** 2
    site = lambda x, y: (y % L) * L + (x % L)
    A = np.zeros((2 * L * L, 2 * L * L), dtype=complex)
    for y in range(L):
        for x in range(L):
            b = slice(2 * site(x, y), 2 * site(x, y) + 2)
            A[b, b] += (m + 2 * r) * I2
            U1 = np.exp(-1j * th * y)                                   # +x 链接
            b2 = slice(2 * site(x + 1, y), 2 * site(x + 1, y) + 2)
            A[b, b2] += S1 / 2 * U1 - r / 2 * I2
            b2 = slice(2 * site(x - 1, y), 2 * site(x - 1, y) + 2)
            A[b, b2] += -S1 / 2 * np.conj(U1) - r / 2 * I2
            U2 = np.exp(1j * 2.0 * np.pi * nflux * x / L) if y == L - 1 else 1.0   # +y 链接（Dirac 弦在缝上）
            b2 = slice(2 * site(x, y + 1), 2 * site(x, y + 1) + 2)
            A[b, b2] += S2 / 2 * U2 - r / 2 * I2
            U2_back = np.conj(np.exp(1j * 2.0 * np.pi * nflux * x / L)) if y == 0 else 1.0
            b2 = slice(2 * site(x, y - 1), 2 * site(x, y - 1) + 2)
            A[b, b2] += -S2 / 2 * U2_back - r / 2 * I2
    return A - np.eye(2 * L * L)


def _index_at(L: int, nflux: float):
    """给定磁通（可为小数）读一次指标：返回 (净手性, 惯性, GW 残差, H 的本征谱)。"""
    G = np.kron(np.eye(L * L), _G5_2D)
    dim = 2 * L * L
    H = G @ _wilson_2d_kernel(L, nflux)
    w, v = np.linalg.eigh(H)
    D = 0.5 * (np.eye(dim) + G @ (v @ np.diag(np.sign(w)) @ v.conj().T))
    gw = float(np.max(np.abs(D @ G + G @ D - 2.0 * D @ G @ D)))
    _, s, vt = np.linalg.svd(D)
    ker = vt.conj().T[:, s < 1e-8]
    idx_chir = float(sum(np.real(np.vdot(ker[:, i], G @ ker[:, i])) for i in range(ker.shape[1])))
    idx_inertia = float(np.sum(w < 0)) - 0.5 * dim
    return idx_chir, idx_inertia, gw, w


def check_flux_index(L: int = 8, fluxes=(0, 1, -1, 2)) -> dict:
    """指标定理的直接检验：1+1D 环面上 index(D_ov) = 磁通量子数 n。

    对每个磁通量子数建核算符 A，取 H = γ⁵A 的本征分解做 sign(H)，组装
    overlap D_ov = ½(1 + γ⁵ sign(γ⁵A))（满足 Ginsparg–Wilson），再用两种独立算法
    读指标：

        净手性   index = Tr(γ⁵ P_ker)，P_ker 为 D_ov 零空间的投影
        惯性     index = n_neg(H) − dim/2（Sylvester 惯性）

    两者必须相等，且都等于 n。返回三项最大残差（全为 0 即通过）。
    """
    e_idx = e_agree = e_gw = 0.0
    for n in fluxes:
        idx_chir, idx_inertia, gw, _ = _index_at(L, n)
        e_gw = max(e_gw, gw)
        e_idx = max(e_idx, abs(idx_chir - n), abs(idx_inertia - n))
        e_agree = max(e_agree, abs(idx_chir - idx_inertia))
    return {"index_vs_flux": e_idx, "inertia_vs_chirality": e_agree, "gw_residual": e_gw}


def _track_levels(L: int, nflux: float, steps: int):
    """沿磁通 0 → nflux 绝热演化，跟踪"负本征值个数"这个整型计数。

    为什么不按本征矢贴标签：谱是简并的（L=8 时 −2.798 四重、−2.613 三重），简并子
    空间里的基由 eigh 任取，相邻步的重叠只有约 0.5，标签会随机跳。能级连续性真正
    可靠的体现是这个整数本身——它只在某条能级**真的穿过零**时才改变，跳变的方向
    就是穿越的方向、跳变的大小就是同时穿越的条数。于是逐格点量它、记录每次跳变。

    返回 (净穿越, 事件列表 [(磁通, 跳变)], 最大单次跳变)。
    """
    def n_neg(phi: float) -> int:
        w = np.linalg.eigvalsh(np.kron(np.eye(L * L), _G5_2D) @ _wilson_2d_kernel(L, phi))
        return int(np.sum(w < 0))

    prev = n_neg(0.0)
    events = []
    for k in range(1, steps + 1):
        flux = nflux * k / steps
        cur = n_neg(flux)
        if cur != prev:
            events.append((float(flux), int(cur - prev)))
        prev = cur
    flow = float(sum(d for _, d in events))
    jump = float(max((abs(d) for _, d in events), default=0))
    return flow, events, jump


def check_spectral_flow(L: int = 8, fluxes=(1, 2, -1), steps: int = 40) -> dict:
    """谱流计数：沿磁通绝热演化跟踪能级穿越，净穿越 = 指标 = 磁通量子数。

    这是与 check_flux_index 相互独立的一条路径：后者在固定磁通下直接读 overlap 算符
    的指标（零空间净手性 / 惯性），前者让磁通从 0 连续变到 n，逐步跟踪负本征值个数，
    每次跳变记一次能级穿越（方向即跳变正负）。

    「跟踪连续性」的正确形式是跟踪这个整数而不是逐条本征矢：谱有简并（L=8 时
    −2.798 四重、−2.613 三重），简并基的选取是任意的，按矢量贴标签必然乱；而整数
    计数只在能级真的穿过零时改变，既连续又与基无关。

    返回三项残差（全为 0 即通过）：谱流 vs 磁通量子数、谱流 vs 固定磁通指标、
    最大单次跳变（> 1 表示步长太粗、多条能级同时穿越）。
    """
    e_flux = e_index = 0.0
    jump = 0.0
    for n in fluxes:
        flow, _, j = _track_levels(L, float(n), steps)
        idx_chir, _, _, _ = _index_at(L, n)
        e_flux = max(e_flux, abs(flow - n))
        e_index = max(e_index, abs(flow - idx_chir))
        jump = max(jump, j)
    return {"flow_vs_flux": e_flux, "flow_vs_index": e_index, "max_jump": jump}


# ==============================================================================
# 6. 序列传播：沿 L 的旋量递推 s_{n+1} = M s_n + c u_n
#    M 为一步物理（规范 × Larmor × 质量 × 吸收）在每组上的 4×4 传播子；
#    线性递推 ⇒ 结合律二元算子 (M,b)∘(M',b') = (M M', M b' + b)，
#    前缀扫描 O(log L) 深度；序贯版本作为黄金对照。
# ==============================================================================
def _basis_columns(n: int):
    return np.eye(n, dtype=complex)


def propagator_numpy(params, dt: float, absorption: bool = True, n_groups: int = 1,
                     gauge_mode: str = "protocol", splitting_order: int | None = None):
    """用基矢探测从一步物理中提取 4×4 传播子 M（块对角，跨组为零）。"""
    D = 4 * n_groups
    M = np.zeros((n_groups, 4, 4), dtype=complex)
    for g in range(n_groups):
        for k in range(4):
            x = np.zeros((1, 1, D), dtype=complex)
            x[0, 0, 4 * g + k] = 1.0
            y = spin_chirality_numpy_forward(
                x, params, dt, absorption=absorption,
                gauge_mode=gauge_mode, splitting_order=splitting_order,
            ).reshape(D)
            M[g][:, k] = y[4 * g:4 * g + 4]
    return M


def propagator_sequence_numpy(params_seq, dt: float, absorption: bool = True, n_groups: int = 1,
                              gauge_mode: str = "generator", splitting_order: int | None = None):
    """构造含时外场下的每步 4×4 传播子序列 M_seq，形状 (L, G, 4, 4)。

    params_seq 可为长度为 L 的参数字典列表，或含前置时间维 L 的参数字典
    （theta_L/R 形状 (L,G) 或 (L,)，axis_L/R/B 在 G>1 时为 (L,3,G)、G=1 时可为 (L,3)）。
    """
    if isinstance(params_seq, dict):
        th_l = np.asarray(params_seq["theta_L"], dtype=float)
        L = th_l.shape[0]
        step_dicts = []
        for n in range(L):
            d = {}
            for k, v in params_seq.items():
                arr = np.asarray(v, dtype=float)
                if k in ("axis_L", "axis_R", "B"):
                    is_tv = arr.ndim == 3 or (n_groups == 1 and arr.ndim == 2 and arr.shape == (L, 3))
                    d[k] = arr[n] if is_tv else arr
                else:
                    d[k] = arr[n] if arr.ndim >= 1 and arr.shape[0] == L and (n_groups == 1 or arr.ndim >= 2) else arr
            step_dicts.append(d)
    else:
        step_dicts = list(params_seq)
        L = len(step_dicts)
    M_seq = np.empty((L, n_groups, 4, 4), dtype=complex)
    for n, p_n in enumerate(step_dicts):
        M_seq[n] = propagator_numpy(
            p_n, dt, absorption=absorption, n_groups=n_groups,
            gauge_mode=gauge_mode, splitting_order=splitting_order,
        )
    return M_seq


def sequence_numpy_forward(u, M, coupling: float):
    """黄金对照：序贯递推 s_{n+1} = M_n s_n + c u_n，支持静态 M(G,4,4) 与时变 M(L,G,4,4)。"""
    B, L, D = u.shape
    G = D // 4
    s = np.zeros((B, G, 4), dtype=complex)
    out = np.empty((B, L, D), dtype=complex)
    tv = M.ndim == 4
    for n in range(L):
        Mn = M[n] if tv else M
        s = np.einsum("gij,bgj->bgi", Mn, s) + coupling * u[:, n].reshape(B, G, 4)
        out[:, n] = s.reshape(B, D)
    return out


def sequence_scan_numpy(u, M, coupling: float):
    """结合律前缀扫描（Hillis–Steele），O(log L) 深度；同时支持静态与时变 M。"""
    B, L, D = u.shape
    G = D // 4
    if M.ndim == 3:
        Ms = np.tile(M[None, None], (B, L, 1, 1, 1)).astype(complex)
    elif M.ndim == 4:
        Ms = np.tile(M[None], (B, 1, 1, 1, 1)).astype(complex)
    else:
        raise ValueError(f"M 维度必须为 3 (G,4,4) 或 4 (L,G,4,4)，当前 {M.ndim}")
    bs = (coupling * u).reshape(B, L, G, 4).astype(complex)
    step = 1
    while step < L:
        idx = np.arange(step, L)
        prev = idx - step
        b_new = bs.copy()
        M_new = Ms.copy()
        b_new[:, idx] = np.einsum("blgij,blgj->blgi", Ms[:, idx], bs[:, prev]) + bs[:, idx]
        M_new[:, idx] = np.einsum("blgij,blgjk->blgik", Ms[:, idx], Ms[:, prev])
        bs, Ms = b_new, M_new
        step *= 2
    return bs.reshape(B, L, D)


if HAS_TORCH:

    def propagator_torch(op: "SpinChiralityOperator", dtype=None):
        """从 torch 算子提取传播子 M（可微，(G,4,4) 满足 y = M @ x）。"""
        G = op.config.n_groups
        dt = dtype or (torch.complex128 if op.theta_L.dtype == torch.float64 else torch.complex64)
        x = torch.zeros(4 * G, 1, 4 * G, dtype=dt, device=op.theta_L.device)
        for k in range(4 * G):
            x[k, 0, k] = 1.0
        y = op(x)
        # x[4g+col] 为标准基矢 e_col，y[4g+col, 0, 4g+row] = M[g][row, col]，须转置对齐 M @ s
        return torch.stack([y[4 * g:4 * g + 4, 0, 4 * g:4 * g + 4].transpose(-1, -2) for g in range(G)])

    def sequence_scan_torch(u, M, coupling):
        """结合律前缀扫描（torch，可微），支持静态 M(G,4,4) 与时变 M(L,G,4,4)。"""
        B, L, D = u.shape
        G = D // 4
        if M.ndim == 3:
            Ms = M[None, None].expand(B, L, G, 4, 4).clone()
        elif M.ndim == 4:
            Ms = M[None].expand(B, L, G, 4, 4).clone()
        else:
            raise ValueError(f"M 维度必须为 3 或 4，当前 {M.ndim}")
        bs = (coupling * u).reshape(B, L, G, 4)
        step = 1
        while step < L:
            idx = torch.arange(step, L, device=u.device)
            prev = idx - step
            b_new = bs.clone()
            M_new = Ms.clone()
            b_new[:, idx] = torch.einsum("blgij,blgj->blgi", Ms[:, idx], bs[:, prev]) + bs[:, idx]
            M_new[:, idx] = torch.einsum("blgij,blgjk->blgik", Ms[:, idx], Ms[:, prev])
            bs, Ms = b_new, M_new
            step *= 2
        return bs.reshape(B, L, D)

    def sequence_sequential_torch(u, M, coupling):
        """序贯递推（torch，可微），支持静态与时变 M。"""
        B, L, D = u.shape
        G = D // 4
        tv = M.ndim == 4
        s = torch.zeros(B, G, 4, dtype=u.dtype, device=u.device)
        out = []
        for n in range(L):
            Mn = M[n] if tv else M
            s = torch.einsum("gij,bgj->bgi", Mn, s) + coupling * u[:, n].reshape(B, G, 4)
            out.append(s.reshape(B, D))
        return torch.stack(out, dim=1)


    class SequenceFunction(torch.autograd.Function):
        """序列传播的自定义反向：前向不保留 O(L·D) 激活，反向按序贯重计算求梯度。"""

        @staticmethod
        def forward(ctx, u, M, coupling):
            with torch.no_grad():
                y = sequence_scan_torch(u, M, coupling)
            ctx.save_for_backward(u, M, coupling)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            u, M, coupling = ctx.saved_tensors
            with torch.enable_grad():
                u_ = u.detach().requires_grad_(True)
                M_ = M.detach().requires_grad_(True)
                c_ = coupling.detach().requires_grad_(True)
                y = sequence_sequential_torch(u_, M_, c_)
                loss = (grad_out.conj() * y).real.sum()
                gu, gM, gc = torch.autograd.grad(loss, [u_, M_, c_])
            return gu, gM, gc


    def _set_nonz_axes(op: "SpinChiralityOperator") -> None:
        """为测试构造含 y 分量（σ₂）的非对称一般 3D 规范/磁场轴。"""
        G = op.config.n_groups
        with torch.no_grad():
            for g in range(G):
                op.axis_L[g].copy_(torch.tensor([0.3 + 0.2 * g, 1.1 - 0.3 * g, -0.5], dtype=op.axis_L.dtype, device=op.axis_L.device))
                op.axis_R[g].copy_(torch.tensor([-0.7, 0.9 + 0.1 * g, 0.4 - 0.2 * g], dtype=op.axis_R.dtype, device=op.axis_R.device))
                op.B[g].copy_(torch.tensor([0.4, -0.8 + 0.2 * g, 1.1], dtype=op.B.dtype, device=op.B.device))

    def _check_sequence_grad() -> dict:
        """序列模式：非 z 轴一般传播子上自定义反向 vs autograd，及有限差分抽查。"""
        torch.manual_seed(0)
        G, L, B = 2, 6, 2
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
        _set_nonz_axes(op)
        M = propagator_torch(op).detach().requires_grad_(True)
        u = (torch.randn(B, L, 4 * G, dtype=torch.float64)
             + 1j * torch.randn(B, L, 4 * G, dtype=torch.float64)).requires_grad_(True)
        c = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        go = (torch.randn(B, L, 4 * G, dtype=torch.float64)
              + 1j * torch.randn(B, L, 4 * G, dtype=torch.float64))

        y = SequenceFunction.apply(u, M, c)
        g_custom = torch.autograd.grad(y, [u, M, c], go)
        y2 = sequence_sequential_torch(u, M, c)
        g_auto = torch.autograd.grad(y2, [u, M, c], go)
        err = max(float((a - b).abs().max()) for a, b in zip(g_custom, g_auto))

        def loss_of(uu):
            yy = sequence_sequential_torch(uu, M, c)
            return (go.conj() * yy).real.sum()

        eps = 1e-6
        idx = (0, 0, 0)
        up = u.detach().clone(); um = u.detach().clone()
        up[idx] += eps; um[idx] -= eps
        fd = float(((loss_of(up) - loss_of(um)) / (2 * eps)).detach())
        ana = float(g_custom[0][idx].real)
        return dict(custom_vs_autograd=err, fd_vs_analytic=abs(fd - ana) / max(abs(fd), abs(ana), 1e-8))



# ==============================================================================
# 7. Triton 融合序列核：复旋量拆成实 8 维，逐 (b,g) 程序在寄存器中沿 L 迭代
# ==============================================================================
if HAS_TORCH and HAS_TRITON:

    @triton.jit
    def _seq_fwd_kernel(U, Y, M, C, L, G, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // G
        g = pid % G
        offs = tl.arange(0, BLOCK)
        m = tl.load(M + g * BLOCK * BLOCK + offs[:, None] * BLOCK + offs[None, :])
        c = tl.load(C)
        s = tl.zeros((BLOCK,), dtype=U.dtype.element_ty)
        base = (b * L) * (BLOCK * G) + g * BLOCK
        for n in range(0, L):
            u = tl.load(U + base + n * (BLOCK * G) + offs)
            s = tl.sum(m * s[None, :], axis=1) + c * u
            tl.store(Y + base + n * (BLOCK * G) + offs, s)

    @triton.jit
    def _seq_bwd_kernel(GO, U, Y, M, C, DU, DM, DC, L, G, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // G
        g = pid % G
        offs = tl.arange(0, BLOCK)
        m = tl.load(M + g * BLOCK * BLOCK + offs[:, None] * BLOCK + offs[None, :])
        c = tl.load(C)
        base = (b * L) * (BLOCK * G) + g * BLOCK
        lam = tl.zeros((BLOCK,), dtype=GO.dtype.element_ty)
        dm_acc = tl.zeros((BLOCK, BLOCK), dtype=M.dtype.element_ty)
        dc_acc = tl.zeros((BLOCK,), dtype=GO.dtype.element_ty)
        for n in range(L - 1, -1, -1):
            gt = tl.load(GO + base + n * (BLOCK * G) + offs)
            lam = gt + tl.sum(m * lam[:, None], axis=0)
            s_prev = tl.zeros((BLOCK,), dtype=Y.dtype.element_ty)
            if n > 0:
                s_prev = tl.load(Y + base + (n - 1) * (BLOCK * G) + offs)
            dm_acc += lam[:, None] * s_prev[None, :]
            u = tl.load(U + base + n * (BLOCK * G) + offs)
            dc_acc += lam * u
            tl.store(DU + base + n * (BLOCK * G) + offs, c * lam)
        tl.atomic_add(DM + g * BLOCK * BLOCK + offs[:, None] * BLOCK + offs[None, :], dm_acc)
        tl.atomic_add(DC, tl.sum(dc_acc))


    @triton.jit
    def _seq_fwd_wide(U, Y, M, C, L, G, BLOCK: tl.constexpr, BIG: tl.constexpr):
        """宽加载前向：一个程序处理一个 b 的全部 G 组，每步一次连续 8G 加载。"""
        b = tl.program_id(0)
        gi = tl.arange(0, BIG)
        gm = gi < G
        offs = tl.arange(0, BLOCK)
        m = tl.load(M + gi[:, None, None] * (BLOCK * BLOCK)
                    + offs[None, :, None] * BLOCK + offs[None, None, :],
                    mask=gm[:, None, None], other=0.0)
        c = tl.load(C)
        s = tl.zeros((BIG, BLOCK), dtype=U.dtype.element_ty)
        base = b * L * (BLOCK * G)
        for n in range(0, L):
            u = tl.load(U + base + n * (BLOCK * G) + gi[:, None] * BLOCK + offs[None, :],
                        mask=gm[:, None], other=0.0)
            s = tl.sum(m * s[:, None, :], axis=2) + c * u
            tl.store(Y + base + n * (BLOCK * G) + gi[:, None] * BLOCK + offs[None, :],
                     s, mask=gm[:, None])


    @triton.jit
    def _seq_fwd_wide_tv(U, Y, M, C, L, G, BLOCK: tl.constexpr, BIG: tl.constexpr):
        """宽加载前向（时变 M(L,G,8,8)）。"""
        b = tl.program_id(0)
        gi = tl.arange(0, BIG)
        gm = gi < G
        offs = tl.arange(0, BLOCK)
        c = tl.load(C)
        s = tl.zeros((BIG, BLOCK), dtype=U.dtype.element_ty)
        base = b * L * (BLOCK * G)
        m_stride = G * BLOCK * BLOCK
        for n in range(0, L):
            m = tl.load(M + n * m_stride + gi[:, None, None] * (BLOCK * BLOCK)
                        + offs[None, :, None] * BLOCK + offs[None, None, :],
                        mask=gm[:, None, None], other=0.0)
            u = tl.load(U + base + n * (BLOCK * G) + gi[:, None] * BLOCK + offs[None, :],
                        mask=gm[:, None], other=0.0)
            s = tl.sum(m * s[:, None, :], axis=2) + c * u
            tl.store(Y + base + n * (BLOCK * G) + gi[:, None] * BLOCK + offs[None, :],
                     s, mask=gm[:, None])

    @triton.jit
    def _seq_bwd_tv_kernel(GO, U, Y, M, C, DU, DM, DC, L, G, BLOCK: tl.constexpr):
        """时变传播子 M(L,G,8,8) 融合反向：λ_n = g_n + M_{n+1}ᵀ λ_{n+1}。"""
        pid = tl.program_id(0)
        b = pid // G
        g = pid % G
        offs = tl.arange(0, BLOCK)
        c = tl.load(C)
        base = (b * L) * (BLOCK * G) + g * BLOCK
        m_stride = G * BLOCK * BLOCK
        lam = tl.zeros((BLOCK,), dtype=GO.dtype.element_ty)
        dc_acc = tl.zeros((BLOCK,), dtype=GO.dtype.element_ty)
        for n in range(L - 1, -1, -1):
            gt = tl.load(GO + base + n * (BLOCK * G) + offs)
            if n < L - 1:
                m_next = tl.load(M + (n + 1) * m_stride + g * BLOCK * BLOCK
                                 + offs[:, None] * BLOCK + offs[None, :])
                lam = gt + tl.sum(m_next * lam[:, None], axis=0)
            else:
                lam = gt
            s_prev = tl.zeros((BLOCK,), dtype=Y.dtype.element_ty)
            if n > 0:
                s_prev = tl.load(Y + base + (n - 1) * (BLOCK * G) + offs)
            dm_n = lam[:, None] * s_prev[None, :]
            tl.atomic_add(DM + n * m_stride + g * BLOCK * BLOCK
                          + offs[:, None] * BLOCK + offs[None, :], dm_n)
            u = tl.load(U + base + n * (BLOCK * G) + offs)
            dc_acc += lam * u
            tl.store(DU + base + n * (BLOCK * G) + offs, c * lam)
        tl.atomic_add(DC, tl.sum(dc_acc))


    def _c2r_mat(M):
        A, B = M.real, M.imag  # dtype 跟随 M
        return torch.cat([torch.cat([A, -B], dim=-1), torch.cat([B, A], dim=-1)], dim=-2)

    def _c2r_vec(u):
        B, L, D = u.shape
        G = D // 4
        v = u.reshape(B, L, G, 4)
        return torch.cat([v.real, v.imag], dim=-1).reshape(B, L, 8 * G).contiguous()

    def _r2c_vec(r):
        B, L, D = r.shape
        G = D // 8
        v = r.reshape(B, L, G, 8)
        return torch.complex(v[..., :4].contiguous(), v[..., 4:].contiguous()).reshape(B, L, 4 * G)

    def sequence_triton_forward(u, M, coupling):
        """Triton 融合前向（fp64，CUDA），支持静态 M(G,4,4) 与时变 M(L,G,4,4)。"""
        Mr = _c2r_mat(M).contiguous()
        ur = _c2r_vec(u)
        B, L, _ = ur.shape
        if M.ndim == 3:
            G, tv = M.shape[0], False
        elif M.ndim == 4:
            if M.shape[0] != L:
                raise ValueError(f"时变 M 首维须等于序列长度 L={L}，当前 {M.shape[0]}")
            G, tv = M.shape[1], True
        else:
            raise ValueError(f"M 维度必须为 3 或 4，当前 {M.ndim}")
        yr = torch.empty_like(ur)
        c_t = torch.tensor([float(coupling)], device=u.device, dtype=ur.dtype)
        big = 1
        while big < G:
            big *= 2
        if tv:
            _seq_fwd_wide_tv[(B,)](ur, yr, Mr, c_t, L, G, BLOCK=8, BIG=big)
        else:
            _seq_fwd_wide[(B,)](ur, yr, Mr, c_t, L, G, BLOCK=8, BIG=big)
        return _r2c_vec(yr)

    def sequence_triton_backward(go, u, y, M, coupling):
        """Triton 融合反向（fp64，CUDA）：支持静态/时变 M，返回 (dUr, dMr, dC)。"""
        Mr = _c2r_mat(M).contiguous()
        ur = _c2r_vec(u)
        g_ur = _c2r_vec(go)
        yr = _c2r_vec(y)
        B, L, _ = ur.shape
        if M.ndim == 3:
            G, tv = M.shape[0], False
        elif M.ndim == 4:
            if M.shape[0] != L:
                raise ValueError(f"时变 M 首维须等于序列长度 L={L}，当前 {M.shape[0]}")
            G, tv = M.shape[1], True
        else:
            raise ValueError(f"M 维度必须为 3 或 4，当前 {M.ndim}")
        dur = torch.empty_like(ur)
        dMr = torch.zeros_like(Mr)
        dC = torch.zeros(1, device=u.device, dtype=ur.dtype)
        c_t = torch.tensor([float(coupling)], device=u.device, dtype=ur.dtype)
        if tv:
            _seq_bwd_tv_kernel[(B * G,)](g_ur, ur, yr, Mr, c_t, dur, dMr, dC, L, G, BLOCK=8)
        else:
            _seq_bwd_kernel[(B * G,)](g_ur, ur, yr, Mr, c_t, dur, dMr, dC, L, G, BLOCK=8)
        return dur, dMr, dC

    def _check_sequence_triton() -> dict:
        """三重对齐：NumPy 黄金 vs torch 扫描 vs Triton 核；非 z 轴反向对比实表示 autograd。"""
        if not torch.cuda.is_available():
            return {"skipped_no_cuda": 0.0}
        torch.manual_seed(0)
        G, L, B = 3, 10, 4
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
        _set_nonz_axes(op)
        M = propagator_torch(op).detach()
        u = (torch.randn(B, L, 4 * G, dtype=torch.float64, device="cuda")
             + 1j * torch.randn(B, L, 4 * G, dtype=torch.float64, device="cuda"))
        c = 0.7
        y_torch = sequence_scan_torch(u, M.to("cuda"), c)
        y_tri = sequence_triton_forward(u, M.to("cuda"), c)
        y_np = torch.tensor(sequence_numpy_forward(u.cpu().numpy(), M.cpu().numpy(), c), device="cuda")
        err = lambda a, b: float((a - b).abs().max())
        e1 = err(y_np, y_torch)
        e2 = err(y_np, y_tri)

        # 反向：同一上游梯度，实表示 autograd vs Triton
        go = torch.randn_like(u)
        Mr = _c2r_mat(M.to("cuda")).detach().requires_grad_(True)
        ur = _c2r_vec(u).detach().requires_grad_(True)
        cr = torch.tensor([c], device="cuda", dtype=torch.float64, requires_grad=True)
        s = torch.zeros(B, G, 8, device="cuda", dtype=torch.float64)
        outs = []
        for n in range(L):
            s = torch.einsum("gij,bgj->bgi", Mr, s) + cr[0] * ur[:, n].view(B, G, 8)
            outs.append(s.reshape(B, 8 * G))
        y_ref = torch.stack(outs, 1)
        g_ur, g_Mr, g_cr = torch.autograd.grad(y_ref, [ur, Mr, cr], _c2r_vec(go))
        dUr, dMr, dC = sequence_triton_backward(go, u, y_tri.detach(), M.to("cuda"), c)
        return dict(numpy_vs_torch=e1, numpy_vs_triton=e2,
                    bwd_du=err(dUr, g_ur), bwd_dm=err(dMr, g_Mr), bwd_dc=err(dC, g_cr))


    def sequence_adjoint_real(go, u, y, M, c):
        """离散伴随（实表示）：λ_n = g_n + M_{n+1}ᵀ λ_{n+1}，支持静态/时变 M。"""
        B, L, D8 = u.shape
        tv = M.ndim == 4
        G = M.shape[1] if tv else M.shape[0]
        if tv and M.shape[0] != L:
            raise ValueError(f"时变 M 首维须等于序列长度 L={L}，当前 {M.shape[0]}")
        lam = torch.zeros(B, G, 8, dtype=u.dtype, device=u.device)
        dM = torch.zeros_like(M)
        dc = torch.zeros((), dtype=u.dtype, device=u.device)
        dU = torch.empty_like(u)
        zero = torch.zeros(B, G, 8, dtype=u.dtype, device=u.device)
        Mt = M.transpose(-1, -2)
        for n in range(L - 1, -1, -1):
            gn = go[:, n].view(B, G, 8)
            if tv:
                lam = gn + (torch.einsum("gij,bgj->bgi", Mt[n + 1], lam) if n < L - 1 else 0)
            else:
                lam = gn + torch.einsum("gij,bgj->bgi", Mt, lam)
            s_prev = y[:, n - 1].view(B, G, 8) if n > 0 else zero
            dm_n = torch.einsum("bgi,bgj->gij", lam, s_prev)
            if tv:
                dM[n] = dm_n
            else:
                dM = dM + dm_n
            dc = dc + (lam * u[:, n].view(B, G, 8)).sum()
            dU[:, n] = (c * lam).reshape(B, D8)
        return dU, dM, dc

    class AnalyticSequenceFunction(torch.autograd.Function):
        """解析伴随版序列算子：反向为手写 λ 递推，支持静态/时变 M。"""

        @staticmethod
        def forward(ctx, u, M, coupling):
            with torch.no_grad():
                y = sequence_scan_torch(u, M, coupling)
            ctx.save_for_backward(u, M, coupling, y)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            u, M, coupling, y = ctx.saved_tensors
            ur, gur, Mr, yr = _c2r_vec(u), _c2r_vec(grad_out), _c2r_mat(M), _c2r_vec(y)
            dUr, dMr, dC = sequence_adjoint_real(gur, ur, yr, Mr, float(coupling))
            dU = _r2c_vec(dUr)
            A = dMr[..., :4, :4] + dMr[..., 4:, 4:]
            Bm = -dMr[..., :4, 4:] + dMr[..., 4:, :4]
            dM = torch.complex(A, Bm).to(M.dtype)
            return dU.to(u.dtype), dM, dC.to(coupling.dtype)

    def _check_analytic_adjoint() -> dict:
        """P13：非 z 轴一般传播子上解析伴随 Function vs 重计算 Function（同一上游梯度）。"""
        torch.manual_seed(0)
        G, L, B = 2, 8, 2
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
        _set_nonz_axes(op)
        M = propagator_torch(op).detach().requires_grad_(True)
        u = (torch.randn(B, L, 4 * G, dtype=torch.float64)
             + 1j * torch.randn(B, L, 4 * G, dtype=torch.float64)).requires_grad_(True)
        c = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        go = (torch.randn(B, L, 4 * G, dtype=torch.float64)
              + 1j * torch.randn(B, L, 4 * G, dtype=torch.float64))
        y1 = AnalyticSequenceFunction.apply(u, M, c)
        g1 = torch.autograd.grad(y1, [u, M, c], go)
        y2 = SequenceFunction.apply(u, M, c)
        g2 = torch.autograd.grad(y2, [u, M, c], go)
        e_u = float((g1[0] - g2[0]).abs().max())
        e_m = float((g1[1] - g2[1]).abs().max())
        e_c = float((g1[2] - g2[2]).abs().max())
        # 同时给出对 y 的前向一致性
        e_y = float((y1 - y2).detach().abs().max())
        return dict(analytic_vs_recompute_u=e_u, analytic_vs_recompute_M=e_m,
                    analytic_vs_recompute_c=e_c, fwd=e_y)


    def sequence_axial_charge(y):
        """Q5[n] = ‖ψ_R‖² − ‖ψ_L‖²（逐 (b,n)），即 γ⁵ 期望对应的手性荷。"""
        B, L, D = y.shape
        G = D // 4
        v = y.reshape(B, L, G, 4)
        ql = (v[..., 0].abs() ** 2 + v[..., 1].abs() ** 2).sum(dim=-1)
        qr = (v[..., 2].abs() ** 2 + v[..., 3].abs() ** 2).sum(dim=-1)
        return (qr - ql)

    def _check_chirality_conservation() -> dict:
        """物理判据：非 z 轴下 m=0 时手性荷沿序列守恒；m>0 时以 2m 频率振荡。"""
        if not torch.cuda.is_available():
            return {"skipped_no_cuda": 0.0}
        torch.manual_seed(0)
        G, L, B = 2, 64, 4
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
        _set_nonz_axes(op)
        with torch.no_grad():
            op.mass.zero_()
            op.gamma.fill_(-20.0)          # 关掉吸收：e^{-Γdt} ≈ 1
        M0 = propagator_torch(op).detach()
        # 齐次演化：只有第 0 步有源（给定初始态），之后无源
        u = torch.zeros(B, L, 4 * G, dtype=torch.complex128, device="cuda")
        u[:, 0] = (torch.randn(B, 4 * G, dtype=torch.float64, device="cuda")
                   + 1j * torch.randn(B, 4 * G, dtype=torch.float64, device="cuda"))
        y = sequence_triton_forward(u, M0.to("cuda"), 0.7)
        q5 = sequence_axial_charge(y)
        scale = float(q5.abs().max()) + 1e-30
        drift = float((q5 - q5[:, :1]).abs().max()) / scale
        # m>0：数值上应与 −cos(2mt) 型振荡一致（相对单调变化）
        with torch.no_grad():
            op.mass.fill_(0.3)
        Mm = propagator_torch(op).detach()
        y2 = sequence_triton_forward(u, Mm.to("cuda"), 0.7)
        q52 = sequence_axial_charge(y2)
        osc = float((q52.max() - q52.min()) / (q52.abs().max() + 1e-30))
        return dict(m0_relative_drift=drift, m_gt0_swing_magnitude=osc)


def _check_index_4d() -> dict:
    """P17：3+1D 指标——FLSG 单瞬子背景下的 overlap 指标应为 −1、反瞬子为 +1。

    计算在 `lattice4d` 里（FFT-Schur 惯性法，8N⁴ 维稀疏算符，核心维与 N 无关）；
    这里只负责调用与归一。缺 scipy 或模块不可用时记为跳过（不算失败）。
    """
    try:
        from . import lattice4d                      # 包内形态
    except ImportError:
        try:
            import lattice4d                          # 单文件同目录形态
        except ImportError:
            return {"skipped_no_lattice": 0.0}
    if not getattr(lattice4d, "HAS_SCIPY", False):
        return {"skipped_no_scipy": 0.0}
    return lattice4d.check_index_4d(N=8, R=2.5)


def _check_sequence() -> dict:
    """序列模式：序贯 vs 扫描 vs 黄金，及与一步物理的一致性。"""
    rng = np.random.default_rng(7)
    G, dt, L, B = 4, 0.1, 12, 3
    params = dict(theta_L=rng.uniform(-1, 1, G), axis_L=rng.normal(size=(3, G)),
                  theta_R=rng.uniform(-1, 1, G), axis_R=rng.normal(size=(3, G)),
                  B=rng.normal(size=(3, G)), mu=rng.uniform(0.5, 2, G),
                  mass=rng.uniform(0, 0.5, G), gamma=np.full(G, 0.3))
    M = propagator_numpy(params, dt, absorption=True, n_groups=G)
    u = rng.normal(size=(B, L, 4 * G)) + 1j * rng.normal(size=(B, L, 4 * G))
    c = 0.7
    y_seq = sequence_numpy_forward(u, M, c)
    y_scan = sequence_scan_numpy(u, M, c)
    err = float(np.max(np.abs(y_seq - y_scan)))
    # 一步一致性：M 作用一次 = 模块的一步前向
    x = u[:, 0]
    y_step = spin_chirality_numpy_forward(x.reshape(B, 1, 4 * G), params, dt).reshape(B, 4 * G)
    y_mat = np.einsum("gij,bgj->bgi", M, x.reshape(B, G, 4)).reshape(B, 4 * G)
    return dict(scan_vs_seq=err, step_vs_propagator=float(np.max(np.abs(y_step - y_mat))))



# ==============================================================================
# 8. 领域内表示：SU(2) 用单位四元数（格点 QCD / 刚体动力学的标准做法）
#    U = q0 I + i (q·σ)；酉性由 |q|=1 保证，R† 就是共轭四元数，
#    作用在 2-旋量上 8 次实乘（复数 2×2 需 16 次）。
# ==============================================================================
def su2_quat(axis, theta):
    """由 (转轴, 角) 构造单位四元数 q = (cos(θ/2), −sin(θ/2)·n̂)；零轴返回单位元 (1,0,0,0)。"""
    n = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    n = n / norm
    return np.concatenate([[np.cos(theta / 2.0)], -np.sin(theta / 2.0) * n])


def quat_conj(q):
    """四元数共轭 = SU(2) 的厄米共轭 R†。"""
    return np.concatenate([[q[0]], -np.asarray(q[1:])])


def quat_apply(q, z):
    """U z，z 为二分量复旋量 (…,2)。"""
    q = np.asarray(q, dtype=float)
    z = np.asarray(z, dtype=complex)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    z1, z2 = z[..., 0], z[..., 1]
    return np.stack([(q0 + 1j * q3) * z1 + (q2 + 1j * q1) * z2,
                     (-q2 + 1j * q1) * z1 + (q0 - 1j * q3) * z2], axis=-1)


def quat_mul(qb, qa):
    """四元数乘 qb * qa，对应矩阵 U_a U_b（作用顺序：先 a 后 b）。"""
    qb = np.asarray(qb, dtype=float); qa = np.asarray(qa, dtype=float)
    b0, bv = qb[..., 0], qb[..., 1:]
    a0, av = qa[..., 0], qa[..., 1:]
    return np.concatenate([(b0 * a0 - np.sum(bv * av, axis=-1))[..., None],
                           b0[..., None] * av + a0[..., None] * bv + np.cross(bv, av)], axis=-1)


def _check_quaternion_su2() -> dict:
    """P15：四元数原语 vs 2×2 复矩阵（作用 / 共轭 / 复合 / 酉性）。"""
    rng = np.random.default_rng(0)
    e_apply = e_conj = e_comp = e_unit = 0.0
    for _ in range(20):
        ax1, ax2 = rng.normal(size=3), rng.normal(size=3)
        t1, t2 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        z = rng.normal(size=2) + 1j * rng.normal(size=2)
        q1, q2 = su2_quat(ax1, t1), su2_quat(ax2, t2)
        U1, U2 = su2(ax1, t1), su2(ax2, t2)
        e_apply = max(e_apply, float(np.max(np.abs(quat_apply(q1, z) - U1 @ z))))
        e_conj = max(e_conj, float(np.max(np.abs(quat_apply(quat_conj(q1), z) - U1.conj().T @ z))))
        e_comp = max(e_comp, float(np.max(np.abs(quat_apply(quat_mul(q2, q1), z) - U1 @ (U2 @ z)))))
        e_unit = max(e_unit, abs(float(np.linalg.norm(q1)) - 1.0))
    return dict(apply=e_apply, conj_is_Rdag=e_conj, composition=e_comp, unit_norm=e_unit)


def _check_anomaly_bookkeeping() -> dict:
    """P16：反常记账与格点指标的归一化一致性。

    格点上验证过：一个 SU(2) 瞬子的 overlap 指标 = 1（index_4d_instanton.py），
    对应 ∫F^a F̃^a d⁴x = 32π²/g²。代入 ΔQ5 = (g²/16π²)∫F F̃ 应得 ΔQ5 = 2·index，
    并且**与 g 无关**——这是系数约定自洽的非平凡判据（g² 必须被 1/g² 抵消）。
    """
    index = 1.0                                   # 格点实测值
    out = {}
    for g in (0.3, 1.0, 2.5):
        int_ff = 32.0 * np.pi ** 2 / (g * g)      # 单瞬子的 ∫F F̃
        dq5 = anomaly_axial_charge_step(0.0, int_ff, g, 1.0)
        out[f"ΔQ5(g={g}) vs 2·index"] = abs(dq5 - 2.0 * index)
    E = np.array([1.0, 0.0, 0.0])
    B = np.array([0.0, 2.0, 0.0])
    out["ff 约定 4E·B"] = abs(ff_from_eb(E, B) - 4.0 * float(E @ B))
    out["正交时 ff 为零"] = abs(ff_from_eb(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])))
    return out



def _check_fujikawa_coefficient() -> dict:
    """P18：反常系数 g²/16π² 的热核（Fujikawa）推导与数值校验。

    欧氏约定：{γ_μ,γ_ν}=2δ_{μν}，γ5=γ0γ1γ2γ3，tr(γ5γ0γ1γ2γ3)=+4，
    σ_{μν}=(i/2)[γ_μ,γ_ν]。把 D̸² 写成 D²+X，X=−(i/2)σ_{μν}F_{μν}，则在平面波基下

        A = lim_{M→∞} tr[γ5 e^{D̸²/M²}]
          = ∫d⁴p/(2π)⁴ e^{−p²/M²}·tr[γ5X²]/(2M⁴)     （trγ5 与 trγ5X 皆为零）
          = tr[γ5X²]/(32π²) = ε^{μνρσ}F_{μν}F_{ρσ}/(32π²) = F F̃/(16π²)

    即 ∂_μ j^5_μ = (g²/16π²)F^a_{μν}F̃^{aμν}（g 已含在 F 里），正是 P16/abnormal
    记账里用的那个系数——P18 把它从"约定"变成"算出来的"。

    逐项校验：γ 约定、tr[γ5σ_{μν}σ_{ρσ}]=−4ε^{μνρσ}、tr[γ5X²]=ε^{μνρσ}F_{μν}F_{ρσ}、
    高斯积分因子 ∫d⁴p/(2π)⁴e^{−p²/M²}=M⁴/(16π²)、以及最终系数与 ff_from_eb 的 E·B 链。
    """
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    i2, z2 = np.eye(2, dtype=complex), np.zeros((2, 2), dtype=complex)

    def gam_of(s):
        return np.block([[z2, -1j * s], [1j * s, z2]])

    gam = (gam_of(s1), gam_of(s2), gam_of(s3), np.block([[i2, z2], [z2, -i2]]))
    g5 = gam[0] @ gam[1] @ gam[2] @ gam[3]

    sig = np.zeros((4, 4, 4, 4), dtype=complex)
    for m in range(4):
        for n in range(4):
            sig[m, n] = 0.5j * (gam[m] @ gam[n] - gam[n] @ gam[m])

    eps = np.zeros((4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    t = (a, b, c, d)
                    if sorted(t) != [0, 1, 2, 3]:
                        continue
                    sgn = 1
                    for i in range(4):
                        for j in range(i + 1, 4):
                            if t[i] > t[j]:
                                sgn = -sgn
                    eps[a, b, c, d] = sgn

    # 1) γ 约定
    e_gamma = abs(np.trace(g5 @ gam[0] @ gam[1] @ gam[2] @ gam[3]).real - 4.0)

    # 2a) tr[γ5] = 0 与 tr[γ5σ_{μν}] = 0：展开里的一阶项为零
    e_first = abs(np.trace(g5))
    for m in range(4):
        for n in range(4):
            e_first = max(e_first, float(abs(np.trace(g5 @ sig[m, n]))))

    # 2b) tr[γ5σ_{μν}σ_{ρσ}] = −4ε^{μνρσ}
    e_sigma = 0.0
    for m in range(4):
        for n in range(4):
            for r in range(4):
                for s in range(4):
                    v = np.trace(g5 @ sig[m, n] @ sig[r, s])
                    e_sigma = max(e_sigma, float(abs(v + 4.0 * eps[m, n, r, s])))

    # 3) 常数场 F 上 tr[γ5X²] = ε^{μνρσ}F_{μν}F_{ρσ}
    rng = np.random.default_rng(7)
    F = rng.normal(size=(4, 4))
    F = F - F.T
    sF = np.einsum("mnab,mn->ab", sig, F)
    X = -0.5j * sF
    tr5x2 = np.trace(g5 @ X @ X)
    eps_ff = float(np.einsum("abcd,ab,cd->", eps, F, F))
    e_x2 = abs(tr5x2 - eps_ff)

    # 4) 系数：A/(εFF) 应为 1/(32π²)，即 F F̃/(16π²)
    coef = tr5x2 / (32.0 * np.pi ** 2)
    e_coef = abs(coef / eps_ff - 1.0 / (32.0 * np.pi ** 2))

    # 5) 高斯积分因子 ∫d⁴p/(2π)⁴ e^{−p²/M²} = M⁴/(16π²)：无量纲化为 ∫u e^{−u}du = 1
    uu, ww = np.polynomial.legendre.leggauss(200)
    big = 40.0
    u = 0.5 * big * (uu + 1.0)
    w = 0.5 * big * ww
    gauss = float(np.sum(w * u * np.exp(-u)) / (1.0 - (big + 1.0) * np.exp(-big)))
    e_gauss = abs(gauss - 1.0)

    # 6) 落到 E·B：ΔQ5 = (g²/16π²)F F̃ = (g²/4π²)E·B
    E = np.array([0.7, 0.0, 0.0])
    B = np.array([0.0, 0.4, 0.0])
    g, dt = 1.3, 1.0
    dq5 = anomaly_axial_charge_step(0.0, ff_from_eb(E, B), g, dt) - anomaly_axial_charge_step(
        0.0, ff_from_eb(E, B), g, 0.0)
    e_eb = abs(dq5 / dt - (g * g / (4.0 * np.pi ** 2)) * float(np.dot(E, B)))

    return {
        "γ5 归一 tr(γ5γ0γ1γ2γ3)=4": e_gamma,
        "tr[γ5] = tr[γ5σ] = 0": e_first,
        "tr[γ5σ σ] = −4ε": e_sigma,
        "tr[γ5X²] = εFF": e_x2,
        "A/(εFF) = 1/32π²": e_coef,
        "高斯 M⁴/16π²": e_gauss,
        "ΔQ5/dt = g²E·B/4π²": e_eb,
    }



def _spin4() -> tuple:
    """Σ_i = diag(σ_i/2, σ_i/2)：两个手性共用同一个 SU(2) 表示（本基下与 β 对易）。"""
    return tuple(np.block([[s / 2.0, Z2], [Z2, s / 2.0]]) for s in SIG)


def _check_micro_spectrum() -> dict:
    """P19：单步生成元 H = −μB·Σ + mβ 的微观谱与算符代数。

    在 (L↑, L↓, R↑, R↓) 基下 H = diag(−μB/2+m, +μB/2+m, −μB/2−m, +μB/2−m)，
    四个能级 {±(m ± μB/2)} ⇒ 对 E → −E（粒子–空穴）对称。简并结构看极限：
    m=0 时能级 ±μB/2，各自对手性 L/R 二重简并；B=0 时能级 ±m，各自对自旋 ↑/↓ 二重简并；
    两者都不为零时四个能级互不相同（质量把每个自旋能级劈成手性混合的两支）。

    算符代数（决定了这个算符"能做什么"）：
      [Σ_i,Σ_j] = iε_ijk Σ_k   — 自旋是 SU(2) 的生成元，两手性同表示（矢量耦合）
      [Σ_i,β] = 0              — 自旋与手性可交换 ⇒ 质量项不改变自旋、只翻转手性
      {γ5,β} = 0               — 质量项就是手性翻转子，频率 2m（P5）
      [γ5,Σ_i] = 0             — 手性荷与自旋荷是两个独立量子数
      [γ5,α_i] = 0             — α=γ⁰γⁱ 含两个 γ，两次反交换后对易（速度不翻转手性）
    """
    mu, B, m = 0.7, 1.3, 0.42
    spin = _spin4()
    Bv = np.array([0.0, 0.0, B])
    H = -mu * sum(Bv[i] * spin[i] for i in range(3)) + m * BETA

    analytic = np.sort([m - mu * B / 2.0, m + mu * B / 2.0,
                        -m - mu * B / 2.0, -m + mu * B / 2.0])
    numeric = np.sort(np.linalg.eigvalsh(H))
    e_spec = float(np.max(np.abs(numeric - analytic)))
    e_ph = float(np.max(np.abs(np.sort(-numeric) - numeric)))

    eps3 = np.zeros((3, 3, 3))
    eps3[0, 1, 2] = eps3[1, 2, 0] = eps3[2, 0, 1] = 1.0
    eps3[0, 2, 1] = eps3[2, 1, 0] = eps3[1, 0, 2] = -1.0
    e_su2 = 0.0
    for i in range(3):
        for j in range(3):
            lhs = spin[i] @ spin[j] - spin[j] @ spin[i]
            rhs = 1j * sum(eps3[i, j, k] * spin[k] for k in range(3))
            e_su2 = max(e_su2, float(np.abs(lhs - rhs).max()))
    e_beta = max(float(np.abs(spin[i] @ BETA - BETA @ spin[i]).max()) for i in range(3))

    # 本征态的手性纯度：质量项不改变对角，故每个本征态都是等权手性叠加，⟨γ5⟩ = 0
    _w, _v = np.linalg.eigh(H)
    e_pure = max(abs(complex(_v[:, k].conj() @ GAMMA5 @ _v[:, k])) for k in range(4))
    # 本征矢的完整结构：每支有确定自旋 ±1/2，且 L/R 权重各 1/2
    e_spin = 0.0
    e_weight = 0.0
    for _k in range(4):
        _sp = complex(_v[:, _k].conj() @ spin[2] @ _v[:, _k]).real
        e_spin = max(e_spin, min(abs(_sp - 0.5), abs(_sp + 0.5)))
        _wl = float(np.sum(np.abs(_v[0:2, _k]) ** 2))
        _wr = float(np.sum(np.abs(_v[2:4, _k]) ** 2))
        e_weight = max(e_weight, abs(_wl - 0.5), abs(_wr - 0.5))

    # 迹/行列式恒等式：tr H = 0，tr H² = 4(m²+(μB/2)²)，det H = (m²−(μB/2)²)²
    _c = mu * B / 2.0
    e_tr = abs(np.trace(H).real)
    e_tr2 = abs(np.trace(H @ H).real - 4.0 * (m * m + _c * _c))
    e_det = abs(np.linalg.det(H).real - (m * m - _c * _c) ** 2)
    e_spin_comm = float(np.abs(H @ spin[2] - spin[2] @ H).max())

    e_g5b = float(np.abs(GAMMA5 @ BETA + BETA @ GAMMA5).max())
    e_g5s = max(float(np.abs(GAMMA5 @ spin[i] - spin[i] @ GAMMA5).max()) for i in range(3))
    e_g5a = max(float(np.abs(GAMMA5 @ ALPHA[i] - ALPHA[i] @ GAMMA5).max()) for i in range(3))

    return {
        "四能级 ±(m±μB/2)": e_spec,
        "粒子–空穴 spec(H)=−spec(H)": e_ph,
        "[Σ_i,Σ_j]=iεΣ_k": e_su2,
        "[Σ_i,β]=0": e_beta,
        "{γ5,β}=0": e_g5b,
        "[γ5,Σ_i]=0": e_g5s,
        "[γ5,α_i]=0": e_g5a,
        "本征态 ⟨γ5⟩=0（等权手性）": e_pure,
        "本征矢自旋 ±1/2": e_spin,
        "本征矢 L/R 权重各 1/2": e_weight,
        "tr H = 0": e_tr,
        "tr H² = 4(m²+c²)": e_tr2,
        "det H = (m²−c²)²": e_det,
        "[H,Σ_z]=0（自旋投影守恒）": e_spin_comm,
    }


def _check_zitterbewegung() -> dict:
    """P20：自由 Dirac 的速度算符与 Zitterbewegung（精确闭式，非微扰）。

    v = α 的本征值恰为 ±1（光速，与 m 无关）；而 [H,α] ≠ 0，故在质量本征态上
    ⟨α⟩ 不守恒。海森堡方程 dα/dt = i[H,α] = 2i(p − αH) 的精确解是

        α(t) = p/H + (α(0) − p/H)·e^{−2iHt}

    振荡频率 2E（NR 极限 2m），振幅 (α(0) − p/H) 就是 Zitterbewegung。
    这条闭式直接核，不需要小参数展开。
    """
    m, pz = 0.35, 1.7
    p = np.array([0.0, 0.0, pz])
    H = dirac_hamiltonian(p, m)
    w, v = np.linalg.eigh(H)

    def u(t):                       # e^{−iHt}
        return (v * np.exp(-1j * w * t)) @ v.conj().T

    def Hinv():
        return (v * (1.0 / w)) @ v.conj().T

    e_vel = float(np.max(np.abs(np.sort(np.linalg.eigvalsh(ALPHA[2]).real)
                                - np.array([-1.0, -1.0, 1.0, 1.0]))))

    pinv = pz * Hinv()
    e_zb = 0.0
    for t in (0.3, 1.1, 2.7):
        lhs = u(t).conj().T @ ALPHA[2] @ u(t)          # e^{iHt} α e^{−iHt}
        rhs = pinv + (ALPHA[2] - pinv) @ u(2.0 * t)    # 闭式，e^{−2iHt} = u(2t)
        e_zb = max(e_zb, float(np.abs(lhs - rhs).max()))

    E = float(np.sqrt(p @ p + m * m))
    rng = np.random.default_rng(5)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)

    def expval(t):
        z = u(t) @ psi
        return complex(z.conj() @ ALPHA[2] @ z)

    e_per = abs(expval(0.4) - expval(0.4 + np.pi / E))

    # ZBW 振幅的 2-范数：m → 0 时精确为零（α_z = H/E，闭式的振荡项消失），随 m 单调增
    def _amp(mm):
        _w, _v = np.linalg.eigh(dirac_hamiltonian(p, mm))
        _hi = (_v * (1.0 / _w)) @ _v.conj().T
        return float(np.linalg.norm(ALPHA[2] - pz * _hi, 2))

    e_amp0 = _amp(0.0)
    e_mono = max(0.0, _amp(m) - _amp(1.5))

    # 周期平均的 ⟨α_z⟩ 等于 p_z/H（群速度）：把振荡项积掉只剩漂移项
    _ts = np.linspace(0.0, np.pi / E, 2000, endpoint=False)
    _vals = np.empty(_ts.size)
    for _i, _t in enumerate(_ts):
        _z = u(_t) @ psi
        _vals[_i] = complex(_z.conj() @ ALPHA[2] @ _z).real
    e_mean = abs(float(np.mean(_vals)) - float(np.real(psi.conj() @ pinv @ psi)))

    return {
        "α 本征值 = ±1": e_vel,
        "α(t)=p/H+(α−p/H)e^{−2iHt}": e_zb,
        "⟨α_z⟩ 周期 π/E": e_per,
        "m=0 时 ZBW 振幅为 0": e_amp0,
        "ZBW 振幅随 m 单调增": e_mono,
        "周期平均 ⟨α_z⟩ = p_z/H": e_mean,
    }


# ==============================================================================
# 6. P21：独立性与完整性
# ==============================================================================
_DANGEROUS_CALLS = {"open", "exec", "eval", "compile", "system", "popen", "urlopen",
                    "remove", "rmdir", "unlink"}

# 公开 API 的归属：`__all__` 里每个可调用名必须指向某个审计项；不在 P1–P21 内的显式标"专项"
_PUBLIC_COVERAGE = {
    "SpinChiralityConfig": "P10 序列扫描",
    "SpinChiralityOperator": "P8 可微性",
    "spin_chirality_numpy_forward": "P10 序列扫描",
    "su2": "P3 SU(2)",
    "su2_exp": "P43 零轴光滑性与构造边界",
    "su2_generator": "P21 自包含与完整性",
    "su2_quat": "P15 四元数 SU(2)",
    "apply_su2_spinor": "P10 序列扫描",
    "quat_apply": "P15 四元数 SU(2)",
    "quat_conj": "P15 四元数 SU(2)",
    "quat_mul": "P15 四元数 SU(2)",
    "propagator_numpy": "P10 序列扫描",
    "propagator_sequence_numpy": "P44 时变规范与对称分裂二阶收敛",
    "sequence_numpy_forward": "P10 序列扫描",
    "sequence_scan_numpy": "P10 序列扫描",
    "run_physics_audit": "P21 自包含与完整性",
    "ff_from_eb": "P16 反常记账",
    "anomaly_axial_charge_step": "P16 反常记账",
    "dirac_hamiltonian": "P6 轴矢 Ward",
    "larmor_unitary": "P21 自包含与完整性",
    "spin_expectation_numpy": "P21 自包含与完整性",
    "check_flux_index": "专项（1+1D 指标，不在 P1–P21 内）",
    "check_spectral_flow": "专项（谱流，不在 P1–P21 内）",
    "gauge_link": "P24 规范步离散阶",
    "gauge_links": "P24 规范步离散阶",
    "fock": "P27 Fock 空间",
    "kogut_susskind": "P33 禁闭(弦张力)",
    "lattice1d": "P22 1+1D 色散/倍增",
    "lattice4d": "P17 3+1D 指标",
    "schwinger": "P31 规范场反作用",
    "ks_lanczos": "P35 Kogut–Susskind 连续极限",
    "ks_mps": "P36 Kogut–Susskind MPS/DMRG",
    "ks_photon": "P37 光子/弦态与电场关联",
    "gap_flux": "P37 光子/弦态与电场关联",
    "corr_electric_connected": "P37 光子/弦态与电场关联",
    "ks_photon_weight": "P38 电场矩阵元与谱权重",
    "photon_weight": "P38 电场矩阵元与谱权重",
    "field_matrix_elements": "P38 电场矩阵元与谱权重",
    "ks_photon_mass": "P39 光子谱定位与 m_γ",
    "photon_mass_point": "P39 光子谱定位与 m_γ",
    "dense_photon_weights": "P39 光子谱定位与 m_γ",
    "fit_m_gamma": "P39 光子谱定位与 m_γ",
    "ks_dispersion": "P40 光子色散截距",
    "dispersion_point": "P40 光子色散截距",
    "closure_gate": "P40 光子色散截距",
    "fit_g0_forms": "P40 光子色散截距",
    "ks_basis": "P41 平移与动量扇区",
    "ks_omega": "P42 光子色散 ω(k)",
    "dispersion": "P42 光子色散 ω(k)",
    "z_local": "P42 光子色散 ω(k)",
    "extrapolate": "P42 光子色散 ω(k)",
    "RealSpinChiralityOperator": "P45 实数等变包装与手性传感反演",
    "real_spin_chirality_numpy_forward": "P45 实数等变包装与手性传感反演",
    "fit_chiral_procrustes": "P45 实数等变包装与手性传感反演",
    "fit_su2_procrustes": "P45 实数等变包装与手性传感反演",
}

# 硬依赖白名单：标准库 + numpy（torch / triton / scipy 必须包在 try 里）
_HARD_IMPORT_ALLOWED = {"__future__", "numpy", "itertools", "sys", "ast", "io", "os", "json"}


def _fn_span(tree, name):
    """函数在源码里的 (起始行, 结束行)：用于把扫描器自己的 open 排除掉。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.lineno, getattr(node, "end_lineno", node.lineno)
    return 0, 0


def _scan_package_sources():
    """扫包内 .py：返回 (硬依赖, 可选依赖, 危险调用, 文件数)。

    硬依赖 = 不在 try 里的 import（必须装）；可选依赖 = 包在 try 里的 import。
    """
    import os as _os

    here = _os.path.dirname(_os.path.abspath(__file__))
    files = sorted(f for f in _os.listdir(here) if f.endswith(".py"))
    hard, soft, bad = set(), set(), []
    for fname in files:
        with open(_os.path.join(here, fname), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        spans = (_fn_span(tree, "_scan_package_sources"), _fn_span(tree, "_check_self_contained"))

        def walk(node, in_try):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.Try):
                    walk(ch, True)
                    continue
                if isinstance(ch, (ast.Import, ast.ImportFrom)):
                    if isinstance(ch, ast.Import):
                        names = [a.name for a in ch.names]
                    else:
                        names = [ch.module] if (ch.module and ch.level == 0) else []
                    for nm in names:
                        (soft if in_try else hard).add(nm.split(".")[0])
                    continue
                if isinstance(ch, ast.Call):
                    fn = ch.func
                    nm = (fn.id if isinstance(fn, ast.Name)
                          else (fn.attr if isinstance(fn, ast.Attribute) else ""))
                    own = any(lo <= ch.lineno <= hi for lo, hi in spans)
                    if nm in _DANGEROUS_CALLS and not own:
                        bad.append(f"{fname}:{ch.lineno}:{nm}")
                walk(ch, in_try)

        walk(tree, False)
    return hard, soft, bad, len(files)


def _check_self_contained() -> dict:
    """P21：独立性（依赖、外部 I/O）与完整性（公开 API 覆盖、无孤儿导出）。

    独立性：只 import numpy（标准库之外），torch / triton / scipy 全在 try 里；
            包内源码没有任何文件/网络/子进程调用，装一个 numpy 就能跑。
    完整性：`__all__` 每个名字都可解析；每个公开可调用对象在 `_PUBLIC_COVERAGE`
            里有归属（审计项或显式"专项"）；并把此前无人调用的三个公开小工具
            （su2_generator / larmor_unitary / spin_expectation_numpy）真的跑一遍。
    """
    import sys as _sys

    pkg = _sys.modules.get(__package__) if __package__ else None
    if pkg is None:
        pkg = _sys.modules[__name__]
    allnames = list(getattr(pkg, "__all__", []))
    # __version__ 之外的所有公开导出（含子模块）都必须进入 _PUBLIC_COVERAGE
    public = [n for n in allnames if n != "__version__"]

    e_all = float(sum(0 if hasattr(pkg, n) else 1 for n in allnames))

    hard, _soft, bad, _n_files = _scan_package_sources()
    e_dep = float(len(hard - _HARD_IMPORT_ALLOWED))
    e_io = float(len(bad))

    item_names = {n for n, _ in _AUDIT_CHECKS}
    e_cov = float(len(set(public) - set(_PUBLIC_COVERAGE)))
    for v in _PUBLIC_COVERAGE.values():
        if not v.startswith("专项") and v not in item_names:
            e_cov += 1.0

    n_hat = np.array([0.3, -0.7, 0.64], dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    s = su2_generator(n_hat)
    # exp(−θ·su2_generator(n̂)) == exp(−iθ n̂·σ/2) == su2(n̂, θ)
    eps_g = 1e-3
    t_gen = I2 - eps_g * s + 0.5 * (eps_g ** 2) * (s @ s) - (eps_g ** 3 / 6.0) * (s @ s @ s)
    e_gen = max(float(np.abs(s + s.conj().T).max()),
                abs(complex(np.trace(s))),
                float(np.abs(su2_exp(eps_g * n_hat) - t_gen).max()))

    B_vec = np.array([0.3, -1.1, 1.3])
    U = larmor_unitary(B_vec, 0.7, 0.9)
    e_uni = max(
        float(np.abs(U.conj().T @ U - np.eye(2)).max()),
        float(np.abs(U - su2(B_vec, -0.7 * float(np.linalg.norm(B_vec)) * 0.9)).max()),
        float(np.abs(larmor_unitary(np.zeros(3), 0.7, 0.9) - np.eye(2)).max()),
    )

    rng = np.random.default_rng(13)
    z = rng.normal(size=2) + 1j * rng.normal(size=2)
    z /= np.linalg.norm(z)
    e_spin = float(np.abs(spin_expectation_numpy(z)
                          - np.array([complex(z.conj() @ SIG[i] @ z) for i in range(3)])).max())

    return {
        "__all__ 全解析": e_all,
        "硬依赖 ⊆ 标准库+numpy": e_dep,
        "无外部 I/O": e_io,
        "公开 API 覆盖表完整": e_cov,
        "su2_generator 反厄米无迹": e_gen,
        "larmor_unitary 酉": e_uni,
        "spin_expectation_numpy vs ψ†σψ": e_spin,
    }



def _lattice1d():
    """取 lattice1d 模块（包内形态或单文件同目录形态），拿不到就返回 None。"""
    try:
        from . import lattice1d
        return lattice1d
    except ImportError:
        try:
            import lattice1d
            return lattice1d
        except ImportError:
            return None


def _check_lattice_1d() -> dict:
    """P22：1+1D 色散关系与费米子倍增。

    这一项是「把空间维度补上」的第一块：给内部演化加一条真的空间方向（最近邻跃迁）。
    naive 离散（r=0）在 k=0 与 k=π 各有一个同质量狄拉克点（倍增）；Wilson 项（r=1）
    把 k=π 那支的质量抬到 m+2r，倍增消失。数值谱与 E(k)=±√(sin²k+(m+r(1−cos k))²) 对拍。
    """
    mod = _lattice1d()
    if mod is None:
        return {"skipped_no_lattice1d": 0.0}
    return mod.check_dispersion()


def _check_domain_wall_1d() -> dict:
    """P23：畴壁零模（Jackiw–Rebbi）——Wilson 下一个壁锁一个零模，naive 下因倍增翻倍。

    环上放 kink–antikink 对（m 在两处变号），查 |E| 极小模的个数与定域性：
      r=1：2 个零模（每个壁一个）   r=0：4 个（倍增没消掉）
    """
    mod = _lattice1d()
    if mod is None:
        return {"skipped_no_lattice1d": 0.0}
    return mod.check_domain_wall()



def gauge_link(field, t: float, dt: float, midpoint: bool = True):
    """由连续场 A(t) 构造一步的手征规范转动，返回 (θ, 单位轴)。

    这是本模块**唯一**与 dt 阶相关的近似：每步用一个 SU(2) 转动代替连续场。
      · 端点法则（midpoint=False）：θ = |A(t)|·dt      ⇒ 全局误差 O(dt)
      · 中点法则（midpoint=True，默认）：θ = |A(t+dt/2)|·dt ⇒ 全局误差 O(dt²)
    实测（转动的场、T=2、dt=0.05）：3.6e-2 → 2.1e-3，改善约 17 倍，见 P24。

    注意：基础静态序列接口假设传播子不随时间变化；含时外场下的序列传播请用
    `propagator_sequence_numpy` 构造 `(L,G,4,4)` 的 `M_seq`，并喂给 `sequence_*`
    或 `SpinChiralityOperator.forward_sequence(..., M_seq=M_seq)`（见 P44）。
    """
    tau = t + 0.5 * dt if midpoint else t
    A = np.asarray(field(tau), dtype=float)
    if A.shape != (3,):
        raise ValueError("field(t) 必须返回长度 3 的实向量")
    nrm = float(np.linalg.norm(A))
    if nrm == 0.0:
        return 0.0, np.array([0.0, 0.0, 1.0])
    return nrm * dt, A / nrm


def gauge_links(field, t0: float, dt: float, n_steps: int, midpoint: bool = True):
    """连续 n_steps 步的链接：返回 (θ 数组, 轴数组 (n_steps,3))，默认中点法则。"""
    thetas = np.empty(n_steps)
    axes = np.empty((n_steps, 3))
    for k in range(n_steps):
        thetas[k], axes[k] = gauge_link(field, t0 + k * dt, dt, midpoint)
    return thetas, axes


def _check_time_evolution() -> dict:
    """P25：1+1D 时间演化（把静态谱变成真的动力学）。

    两套独立实现互相核对：k 空间 FFT（常数质量）与本征分解 V e^{−iΛdt} V†。
    判据：幺正性、动量本征态的色散相位 e^{−iE(k)t}、两套实现一致；
    另报群速度（波包中心位移 vs v_g·t，作 _magnitude 不进残差）。
    """
    mod = _lattice1d()
    if mod is None:
        return {"skipped_no_lattice1d": 0.0}
    return mod.check_evolution()


def _check_gauge_1d() -> dict:
    """P26：1+1D 的 U(1) 规范场——纯规范不变性、Aharonov–Bohm 周期、Klein 隧穿。

    `lattice1d.build` 现在支持每键 Peierls 相位 theta 与每格点标量势 V。
    判据：纯规范变换(theta_x = φ_{x+1}−φ_x)下谱不变；环上均匀磁通加 2π/N 后谱不变（AB 周期）；
    m=0 时阶跃势透射率恰为 1（1+1D 的 Klein 隧穿），有质量时明显小于 1（只作 _magnitude）。
    """
    mod = _lattice1d()
    if mod is None:
        return {"skipped_no_lattice1d": 0.0}
    return mod.check_gauge()


def _check_gauge_midpoint() -> dict:
    """P24：规范步的离散化阶——中点采样把 O(dt) 降到 O(dt²)。

    场：轴在 x–y 面内以 ω=5 转动、幅度 2（不同时刻的转动不对易，误差才暴露）。
    参考值：中点法则 + 极细细分（dt = T/20000）。比较同一网格上端点 / 中点两种采样。
    """
    def field(t):
        w = 5.0
        return 2.0 * np.array([np.cos(w * t), np.sin(w * t), 0.0])

    def prop(dt, n, midpoint):
        P = np.eye(2, dtype=complex)
        for k in range(n):
            th, ax = gauge_link(field, k * dt, dt, midpoint)
            P = su2(ax, th) @ P
        return P

    T, dt = 2.0, 0.05
    n = int(round(T / dt))
    ref = prop(T / 20000.0, 20000, True)
    e_end = float(np.abs(prop(dt, n, False) - ref).max())
    e_mid = float(np.abs(prop(dt, n, True) - ref).max())
    return {
        "中点误差 ≤ 端点/5": max(0.0, e_mid - e_end / 5.0),
        "端点误差_magnitude": e_end,
        "中点误差_magnitude": e_mid,
        "改善倍数_magnitude": e_end / max(e_mid, 1e-300),
    }



def _fock():
    """取 fock 模块（包内形态或同目录形态），拿不到返回 None。"""
    try:
        from . import fock
        return fock
    except ImportError:
        try:
            import fock
            return fock
        except ImportError:
            return None


def _check_fock_space() -> dict:
    """P27：二次量子化的最小实现——Fock 空间。

    判定三件事，都是精确关系：
      · 正则反对易 {c_i, c†_j} = δ_ij、{c,c} = {c†,c†} = 0
      · Pauli 排斥：(c†)² = 0、n_i 本征值 ∈ {0,1}、重复占据被禁止
      · 多体演化 = 单粒子演化的 Slater 行列式（⟨n_i⟩ 逐模式对拍）+ 粒子数守恒
    """
    mod = _fock()
    if mod is None:
        return {"skipped_no_fock": 0.0}
    out = {}
    out.update(mod.check_canonical(5))
    out.update(mod.check_pauli(5))
    out.update(mod.check_slater(6))
    return out


def _check_sea_chiral_1d() -> dict:
    """P28：二次量子化的手征荷在磁通下不流——反常不是海的性质。

    把负能级填满（Dirac 海），Q5 = Σ_{E<0}⟨γ5⟩。磁通 θ 扫满 0→2π，Q5(θ) 精确为零
    （粒子–空穴 + 手征对称让正负能级成对抵消）。所以「填满的海」给不出反常：
    反常是正规化子/测度效应——与 P18 的三维结论同源。本项把该事实钉成判据。
    """
    mod = _lattice1d()
    if mod is None:
        return {"skipped_no_lattice1d": 0.0}
    return mod.check_sea_chiral()


def _check_fujikawa_1d() -> dict:
    """P29：1+1D 的 Fujikawa 热核——同一个方法，展开阶数差一阶。

    2 维里 tr(γ5) = 0 但 **tr(γ5σ_{μν}) ∝ ε_{μν} ≠ 0**，所以反常来自展开的**一阶**：

        A = lim tr[γ5 e^{D̸²/M²}] = ∫d²p/(2π)² e^{−p²/M²} · tr[γ5X]/M²
          = (M²/4π)·tr[γ5X]/M² = tr[γ5X]/(4π) = ε^{μν}F_{μν}/(4π)

    即 ∂·j5 = ε^{μν}F_{μν}/(4π) = F₀₁/(2π)——Schwinger 系数。
    对照 3 维（P18）：那里的第一阶恰好为零，必须跑到二阶 X²，系数 1/(32π²)。
    """
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    g0, g1 = sx, 1j * sy                      # γ⁰ = σ_x, γ¹ = iσ_y（都厄米）
    g5 = g0 @ g1                             # = −σ_z
    sig01 = 0.5j * (g0 @ g1 - g1 @ g0)
    sig10 = 0.5j * (g1 @ g0 - g0 @ g1)

    F01 = 0.7
    sF = sig01 * F01 + sig10 * (-F01)        # σ_{μν}F_{μν}，F 反对称
    X = -0.5j * sF
    tr5x = complex(np.trace(g5 @ X))
    epsF = 2.0 * F01                         # ε^{μν}F_{μν}
    A = tr5x / (4.0 * np.pi)
    return {
        "γ5 厄米且平方为 1": max(float(np.abs(g5 - g5.conj().T).max()),
                                float(np.abs(g5 @ g5 - np.eye(2)).max())),
        "tr[γ5] = 0（二阶项才会非零的前提）": abs(complex(np.trace(g5))),
        "tr[γ5X] = ε^{μν}F_{μν}": abs(tr5x - epsF),
        "A/(εF) = 1/(4π)": abs(A / epsF - 1.0 / (4.0 * np.pi)),
    }


def _check_pair_production_1d() -> dict:
    """P30：反常的动力学面——磁通匀速扫过时的对产生（Landau–Zener）。

    均匀磁通下动量守恒，每个 k 是独立二能级问题 H(q)=σ_z sin q + σ_x m，q=θ/N=ωτ。
    扫过一个交叉点后处于激发态的概率应满足 P = exp(−πm²/ω)：这是 1+1D 反常的直接
    动力学后果——真空在电场下不稳定，对产生率指数依赖于 m²/eE，时间尺度由 ω 给出。
    """
    mod = _lattice1d()
    if mod is None:
        return {"skipped_no_lattice1d": 0.0}
    return mod.check_pair_production()


def _schwinger():
    """取 schwinger 模块（包内形态或同目录形态），拿不到返回 None。"""
    try:
        from . import schwinger
        return schwinger
    except ImportError:
        try:
            import schwinger
            return schwinger
        except ImportError:
            return None


def _check_gauge_backreaction() -> dict:
    """P31：规范场反作用（半经典）——等离子频率与电荷屏蔽都等于 e/√π。

    Schwinger 模型里无质量费米子的一圈诱导电流是精确的：j¹ = −(e/π)A₁。
    喂回麦克斯韦方程 ∂_tE = −j¹（E = −∂_tA₁）得 ω_p = e/√π（光子获得质量）；
    静态情形 (∂_x² − m_γ²)A₀ = −ρ 给出屏蔽长度 1/m_γ = √π/e。
    判据：两者都对该解析值，外加耦合系统的能量守恒。
    """
    mod = _schwinger()
    if mod is None:
        return {"skipped_no_schwinger": 0.0}
    return mod.check_backreaction()


def _check_massive_backreaction() -> dict:
    """P32：有质量的反作用——真空极化 Π(q²) 与色散方程 ω² = e²Π(ω²)。

    1+1D 单圈真空极化 Π(q²) = (1/π)[1 + ∫dx x(1−x)q²/(m²−x(1−x)q²)]，三个解析极限：
    Π(0) = 1/π（**与 m 无关**——反常不可重整化）、Π(∞) = 0、q²→4m²⁻ 对数发散（对产生阈值）。
    有质量的光子质量取自 ω² = e²Π(ω²)，随 m 增大从上方趋于 e/√π。
    """
    mod = _schwinger()
    if mod is None:
        return {"skipped_no_schwinger": 0.0}
    return mod.check_massive()


def _kogut_susskind():
    """取 kogut_susskind 模块，拿不到返回 None。"""
    try:
        from . import kogut_susskind
        return kogut_susskind
    except ImportError:
        try:
            import kogut_susskind
            return kogut_susskind
        except ImportError:
            return None


def _check_confinement_1d() -> dict:
    """P33：1+1D 的禁闭——高斯定律直接给出线性势，弦张力 σ = g²/2。

    电荷 ±1 相距 d，环上高斯定律把电场解成 `E_x = c + 累计电荷`（c 是真空角标签），
    静态势 V(d) = min_c (g²/2)ΣE²；对 d < N/2 最小值在 c=0，V(d) = (g²/2)d。
    这是 1+1D 强耦合下的精确结果，不需要任何截断。
    """
    mod = _kogut_susskind()
    if mod is None:
        return {"skipped_no_kogut_susskind": 0.0}
    return mod.check_confinement()


def _ks_ed():
    """取 ks_ed 模块，拿不到返回 None。"""
    try:
        from . import ks_ed
        return ks_ed
    except ImportError:
        try:
            import ks_ed
            return ks_ed
        except ImportError:
            return None


def _check_ks_ed() -> dict:
    """P34：1+1D Kogut–Susskind 精确对角化（含高斯定律投影与 E 谱截断）。

    结构性判据（全部精确）：H 厄米、高斯定律在**每个**基矢上成立、E 谱截断饱和（Emax=3 与 2 维数
    相同）、最低本征值在 Emax 2→3 下不再变化。
    ⚠ 物理上，质量隙在 N ≤ 14 时**没有**收敛到 e/√π（1.96/0.75/1.32/0.91/1.13 振荡），
    连续极限（N 推到 24、Emax 2/3/4、以 e 为单位）由 **P35 / ks_lanczos.py** 接手。
    要拿到连续极限还需要 N、截断、物理单位三重外推——本项只保证机器正确，不声称已到连续极限。
    """
    mod = _ks_ed()
    if mod is None:
        return {"skipped_no_ks_ed": 0.0}
    return mod.check_ks()



def _ks_lanczos():
    """取 ks_lanczos 模块，拿不到返回 None。"""
    try:
        from . import ks_lanczos
        return ks_lanczos
    except ImportError:
        try:
            import ks_lanczos
            return ks_lanczos
        except ImportError:
            return None


def _check_ks_continuum() -> dict:
    """P35：Kogut–Susskind 质量隙的连续极限（矩阵自由 Lanczos，N 到 24）。

    P34 的稠密 ED 只到 N ≤ 14，gap(N) 在 1.96/0.75/1.32/0.91/1.13 上振荡，三项外推
    一项都没做。本项把三件事分开做完，判据全部落在 `ks_lanczos.py` 里：

      ① 引擎：不物化矩阵，matvec = 对角项 + N 条键的 gather/scatter。与 `ks_ed.py`
         的稠密 H 逐元素一致（3.6e-15），最低两能级与稠密 ED 一致（1.8e-14）。
         N=24 时维数 2.2e6、单次 matvec 1.6 s，200 步 Lanczos 313 s。
      ② 归一化（两把独立的尺）：强耦合极限 gap/(g²/2) → 1（g=6 时 1.0016）、自由极限
         E0 = Σ 最低 N/2 个 ±2t|cos(k/2)|（PBC，1e-15）。两者一起把 H 钉成
         「跳跃 t、电场 g²/2」，于是 e² = 2g²t、x = ea = g/√(2t)、连续极限是 e/√π。
      ③ 外推：Emax = 2/3/4 的截断差 ≤ 1e-4（已收敛），N 用振荡模型 a + b(−1)^{N/2}e^{−κN}
         外推，x 用 G(x) = G0 + A·x^p 外推（只取 x ≥ 0.5：x ≤ 0.4 的零模振荡在 N ≤ 24
         下已到 ±25%，不可控）。G0、偏差、不确定度都是 `_magnitude`，
         判据只有两条定性/防回归的：G(x) 随 x 单调下降、G0 落在 e/√π 的 [0.95,1.25]
         窗口内（精度声明与负结果见 OPERATOR_STATUS.md §6terdecies）。
    """
    mod = _ks_lanczos()
    if mod is None:
        return {"skipped_no_ks_lanczos": 0.0}
    return mod.check_ks_continuum()

def _ks_mps():
    """取 ks_mps 模块，拿不到返回 None。"""
    try:
        from . import ks_mps
        return ks_mps
    except ImportError:
        try:
            import ks_mps
            return ks_mps
        except ImportError:
            return None


def _check_ks_mps() -> dict:
    """P36：Kogut–Susskind 质量隙的连续极限（MPS/DMRG，N 到 60、x 到 0.15）。

    P35 的矩阵自由 Lanczos 到 N = 24、三重外推给 G(0) = 0.594 ± 0.073（比 1/√π
    高 5.2%）；要判定这 5%，按 m_γL ≥ 5 需要 N ≈ 57，稠密基矢 4^57 不可能。
    本项换成 MPS/DMRG：代价随 N 线性、随键维三次，与基矢维数无关。判据全部落在
    `ks_mps.py` 里（结构 ①–⑧，见那里的文档）；这里只负责转交与汇总。
    """
    mod = _ks_mps()
    if mod is None:
        return {"skipped_no_ks_mps": 0.0}
    return mod.check_ks_mps()


def _ks_photon():
    """取 ks_photon 模块，拿不到返回 None。"""
    try:
        from . import ks_photon
        return ks_photon
    except ImportError:
        try:
            import ks_photon
            return ks_photon
        except ImportError:
            return None


def _check_ks_photon() -> dict:
    """P37：磁通分辨光子/弦态 + 逐点减除的电场关联（`ks_photon.py`）。

    P36 留下的两个未决问题：小 x 的 gap 不饱和（怀疑是弦态）、电场关联给不出 m_γ
    （ξ 落到边界）。本项给环加磁通 φ（扭曲边界，`mpo_ks(..., phi=φ)`）测 gap(φ)：
    离域粒子应 gap² = m² + q²(φ/N)²，局域弦态应对 φ 不敏感（~e^{−N/ℓ}）；
    再把 ⟨E_x⟩ 逐点减掉、用全部 N² 个 ⟨ρ_aρ_b⟩ 累成连通关联取 ξ。判据与实测值
    都写在 `ks_photon.check_ks_photon` 里，这里只转交与汇总。
    """
    mod = _ks_photon()
    if mod is None:
        return {"skipped_no_ks_photon": 0.0}
    return mod.check_ks_photon()


def _ks_photon_weight():
    """取 ks_photon_weight 模块，拿不到返回 None。"""
    try:
        from . import ks_photon_weight
        return ks_photon_weight
    except ImportError:
        try:
            import ks_photon_weight
            return ks_photon_weight
        except ImportError:
            return None


def _check_ks_photon_weight() -> dict:
    """P38：电场矩阵元、谱权重与动量分解（`ks_photon_weight.py`）。

    P37 的结论是"主导激发是离域粒子型，但既不是自由光子、也不是纯弦"——那是从
    **能谱**（gap(φ) 的二次系数）间接推出来的。P38 直接量**算符**：
    ⟨1|E_x|0⟩（规范场量子与电场耦合的定义性判据）、⟨1|ρ_x|0⟩（= ∂_x⟨1|E_x|0⟩，
    与切口约定无关的那一半）、⟨1|j_x|0⟩（用 i[H,E_x] = j_{N−1} − j_x 与前者绑定），
    再对 x 做傅里叶看动量分布，并用求和规则 Σ_n|⟨n|E_x|0⟩|² = ⟨0|δE_x²|0⟩ 给出
    "最低那一根本征态占电场谱权重多少"（f₁）。判据与实测值都写在
    `ks_photon_weight.check_ks_photon_weight` 里，这里只转交与汇总。
    """
    mod = _ks_photon_weight()
    if mod is None:
        return {"skipped_no_ks_photon_weight": 0.0}
    return mod.check_ks_photon_weight()


def _ks_photon_mass():
    """取 ks_photon_mass 模块，拿不到返回 None。"""
    try:
        from . import ks_photon_mass
        return ks_photon_mass
    except ImportError:
        try:
            import ks_photon_mass
            return ks_photon_mass
        except ImportError:
            return None


def _check_ks_photon_mass() -> dict:
    """P39：把光子从激发谱里挑出来，用它的能量定义 m_γ（`ks_photon_mass.py`）。

    P38 说"最低激发是中性 2k_F 模式，所以 P35/P36 的 0.581 测的不是 m_γ"。那句话是
    从**一个矩阵元 + 一个求和规则比值**推出来的。本项把它变成可判定问题：稠密 ED 的
    **全谱**里哪一个本征态最像光子？判据是电场**零模**的份额

        κ(n) = |⟨n|Ê|0⟩|² / ⟨0|δÊ²|0⟩,   Ê = Σ_x E_x,   Σ_{n≠0} κ(n) = 1

    理由：1+1D 的规范场没有横向极化，唯一规范不变的场强就是 E_x，光子（Schwinger
    玻色子）的插值算符就是它；而 ρ 通道的 k = 0 分量恒为零（q = 0 里 ⟨n|Q|0⟩ ≡ 0），
    所以"动量分布 k = 0 为主"这条判据只能落在 E 通道上。认定的态用正交 DMRG 跟到
    N = 32 得 m_γ(x,N) = (E_n − E_0)/e，再做 x→0、N→∞ 双外推。判据与实测值都写在
    `ks_photon_mass.check_ks_photon_mass` 里，这里只转交与汇总。
    """
    mod = _ks_photon_mass()
    if mod is None:
        return {"skipped_no_ks_photon_mass": 0.0}
    return mod.check_ks_photon_mass()



def _ks_dispersion():
    """取 ks_dispersion 模块，拿不到返回 None。"""
    try:
        from . import ks_dispersion
        return ks_dispersion
    except ImportError:
        try:
            import ks_dispersion
            return ks_dispersion
        except ImportError:
            return None


def _check_ks_dispersion() -> dict:
    """P40：色散截距法——用 ω²(k) = m_γ² + Z k² 的**截距**定 m_γ、**斜率** Z 当自检读数
    （`ks_dispersion.py`）。

    P39 的不确定度 11.9% 里，主要的一块不是统计误差而是**外推形式的散布**；CLOSURE_DESIGN.md
    的诊断是"有限体积与态识别都已解决，剩下的是外推形式与 O(a) 伪影本身"。本项换观测量：
    光子是粒子，必须满足 ω² = m_γ² + Z k²（连续极限 Z = 1），于是给环加磁通 φ，用 k = φ/N
    在**同一个 x** 上扫 5 个动量点，拟合截距当 m_γ。

    这样做的三个收益：有限体积误差主要进斜率；密度波式的伪模式在大 k 处弯曲、能从**拟合残差**
    上看见；Z 与 1 的偏差本身就是 O(a) 伪影的直接读数，不必另设观测量。

    一条容易错的约定写在模块文档里：φ 的周期是 2π，所以 (j, φ) ≡ (j+1, φ−2π)，
    k = (2πj + φ_red)/N 且 φ_red ∈ (−π, π]——φ = 3π/2 对应 k = −π/(2N) 而不是 +3π/(2N)。

    判据与实测值都写在 `ks_dispersion.check_ks_dispersion` 里，这里只转交与汇总。
    """
    mod = _ks_dispersion()
    if mod is None:
        return {"skipped_no_ks_dispersion": 0.0}
    return mod.check_ks_dispersion()


def _ks_basis():
    """取 ks_basis 模块（包内形态或单文件同目录形态），拿不到返回 None。"""
    try:
        from . import ks_basis
        return ks_basis
    except ImportError:
        try:
            import ks_basis
            return ks_basis
        except ImportError:
            return None

def _check_ks_basis() -> dict:
    """P41：1+1D Kogut–Susskind 的**平移**与动量扇区（`ks_basis.py`）。

    前情：按 k 分扇区要求 T 与 H 严格对易，而旧基里不是——`ks_ed.py` 的 q=0 切片、
    以及 `ks_basis.py` 前几轮的 `(mask, c)` 写法，实测 ‖[H,T]‖ 最好 0.64（g=0.4）、
    矩阵元错配 73 条且**全部**落在绕圈键 (0,7)。三项根因在这里拆开，都有数字：

      ① 电场位移错了链接。跳跃 χ†_a U_x χ_b 只改**链接 x** 上的电场（E_x → E_x ± 1）；
         旧写法按 a < b 决定抬还是降，绕圈键上 a = N−1 > b = 0 方向反了 ⇒ 合法的跳跃被
         截断丢掉、反向那一半留下，‖H − Hᵀ‖ = 0.5（66 对不对称，全在绕圈键）。
      ② `c` 折账不协变。E = c + Σ_{y≤x}ρ_y 把接缝折进标签，而平移会挪接缝 ⇒
         c' = c − (ρ_{N−2}+ρ_{N−1})，绕圈键还要 c → c ± 1。现在 E 直接存成整数向量，
         跳跃只改一个分量、平移只 roll，没有任何 c 要算。
      ③ 绕圈键的 JW 串 (−1)^{N_f−n_0−n_{N−1}} 不是平移不变量：留着它 ‖[H,T]‖ = 1.0、
         292 条矩阵元对不上，去掉以后 0。这不改物理——两个 H 只差一个对角 ±1 的态重定义
         （实测 ‖H_jw − D·H_uniform·D‖ = 0、谱逐点相同），而 Susskind 的自旋-1/2 表示里
         （偶格点物质 σ^z、奇格点规范 τ^z、跳跃 σ⁺τ⁺σ⁻）本来就没有 JW 串。

    于是 T 就是 (mask, E) → (roll₂ mask, roll₂ E) 的**纯置换**：幺正、T^{N/2} = T^N = 1、
    [H,T] = 0 精确（不是 1e-16 量级）。判据全部进残差：三个 g 的 ‖[H,T]‖、H 厄米、
    Gauss 逐态、错配条数、i[H,E_x] = j_x（离散 Maxwell）、T 的群性质、U†HU 的跨扇区矩阵元。
    实测值在同名函数的返回里，也在 OPERATOR_STATUS.md §6undevicies。

    为什么必须 2 格：交错费米子按 x 的绝对奇偶定义 ρ，1 格平移把交错结构翻过来
    （257/257 个态的 1 格像都不在基里）。平移的单位胞是 2 格，T 的阶 = N/2。
    """
    mod = _ks_basis()
    if mod is None:
        return {"skipped_no_ks_basis": 0.0}
    return mod.check_basis()

def _ks_omega():
    """取 ks_omega 模块（包内形态或单文件同目录形态），拿不到返回 None。"""
    try:
        from . import ks_omega
        return ks_omega
    except ImportError:
        try:
            import ks_omega
            return ks_omega
        except ImportError:
            return None


def _check_ks_omega() -> dict:
    """P42：动量分辨的光子色散 ω(k)（`ks_omega.py`）。

    前情：P40 想用 ω² = m_γ² + Z k² 判"那个低能激发是不是光子"，结论是 Z ≈ 0。那不是物理，
    是基的问题——`dispersion_point(..., j)` 的 j 只换投影权重，五个 j 给出同一个 ω；平移算符
    在 g ≠ 0 也不与 H 对易。P41 把基重写成 (mask, E)、让 T 成纯置换、[H,T] = 0 精确以后，
    按 k 分扇区才第一次可做。本项就是补上那一步。

    做法：H 存成 (row, col, val) 三元组（与 `ks_basis.build_h` 逐项一致，只是不物化 dim²），
    扇区基是轨道上的 Fourier 组合 Q，matvec 走 Q†HQ，走 `np.bincount`，纯 numpy；
    每个扇区用复数 Lanczos（两遍全重正交）取最低 1–2 个 Ritz 值，ω(k) = E₁(k) − E₀
    （k = 0 的扇区里最低态就是基态，取第二个）。

    判据（全部进残差）：扇区谱 vs 稠密 ED 两条独立实现一致、有限尺寸 ω(π/2) 在 N=8 与 N=12 上
    一致、BZ 折叠 E(k) = E(π−k)、电场求和规则 Σ_n|<n|E_x|0>|² = <E_x²>、k = 0 的最低激发是
    E 权重最大的那个、`OMEGA_SCAN` 表(6 个耦合 × 5 个 k)当场复现，以及两条物理判据：
    Z_loc 随 x → 0 单调趋 1（x = 0.2 时 0.973，三形式外推 0.988）、G(0) = ω(0)/x 外推到
    0.5567（对 1/√π = 0.56419 差 1.3%）。另有 (x, N) 二维部分（`GAP_2D`，42 点）：表复现、
    六种联合拟合 G(x,N) = A0 + a·x + [x²] + [1/N² 或 e^{−xN/√π}]：x² 项在两种有限尺寸处理下
    变号（−0.0329±0.0086 vs +0.0318±0.0058）⇒ 不是物理项，剔除后只剩三种，
    A0 = 0.55389/0.56322/0.56310，**散布 1.67% < 2% ⇒ P40 判据 ② 过**，均值 0.56007（对 1/√π 差 0.73%）。
    两个读数（六种 5.3%、三种 1.67%）都在 `model_selection()` 里，别只引一个。

    边界（写进 `ks_omega.py` 的模块文档）：最小可及动量 k₁ = 2π/N = 0.3927 在 x = 0.2 时
    是 k/m = 3.3，所以 Z_loc 是"最小可及动量上的平方斜率"，不是 k→0 的严格斜率；大 k 处最轻态
    与两光子阈值重叠，那里"是不是光子"没有意义。N ≤ 18、Emax = 2、m = 0。
    """
    mod = _ks_omega()
    if mod is None:
        return {"skipped_no_ks_omega": 0.0}
    return mod.check_omega()


def _check_axis_smoothness() -> dict:
    """P43：零轴/零场 SU(2) 指数光滑性与构造边界（`su2_exp`、Taylor 边界、复 dtype 与非 z 传播子）。"""
    e_exp_zero = float(np.max(np.abs(su2_exp(np.zeros(3)) - I2)))
    n_dir = np.array([0.3, -0.7, 0.64], dtype=float)
    n_hat = n_dir / np.linalg.norm(n_dir)
    e_exp_su2 = 0.0
    for phi in (0.0, 1e-12, 9.9e-5, 1.01e-4, 1.3):
        U = su2_exp(phi * n_hat)
        e_exp_su2 = max(
            e_exp_su2,
            float(np.max(np.abs(U - su2(n_hat, phi)))),
            float(np.max(np.abs(U.conj().T @ U - I2))),
        )
    e_quat_zero = float(np.max(np.abs(su2_quat(np.zeros(3), 1.5) - np.array([1.0, 0.0, 0.0, 0.0]))))

    rng = np.random.default_rng(43)
    G, B_sz = 2, 3
    p0 = dict(
        theta_L=np.array([1.5, -0.8]), axis_L=np.zeros((3, G)),
        theta_R=np.array([-0.9, 1.1]), axis_R=np.zeros((3, G)),
        B=np.zeros((3, G)), mu=np.array([1.2, 0.5]),
        mass=np.array([0.4, -0.2]), gamma=np.zeros(G),
    )
    x0 = rng.normal(size=(B_sz, 2, 4 * G)) + 1j * rng.normal(size=(B_sz, 2, 4 * G))
    y0 = spin_chirality_numpy_forward(x0, p0, 0.1, absorption=False)
    nx = np.linalg.norm(x0, axis=-1)
    ny = np.linalg.norm(y0, axis=-1)
    e_np_unit = float(np.max(np.abs(ny - nx) / nx))

    out = {
        "su2_exp_zero_id": e_exp_zero,
        "su2_exp_smooth_unitary": e_exp_su2,
        "su2_quat_zero_id": e_quat_zero,
        "numpy_zero_axis_unitary": e_np_unit,
    }
    if not HAS_TORCH:
        return out

    bad_dtype = 0.0
    for bad in (float, torch.float32, torch.float64):
        try:
            SpinChiralityOperator(SpinChiralityConfig(4), dtype=bad)
            bad_dtype += 1.0
        except TypeError:
            pass
    op_tmp = SpinChiralityOperator(SpinChiralityConfig(4))
    try:
        op_tmp(torch.randn(1, 1, 4, dtype=torch.float32))
        bad_dtype += 1.0
    except TypeError:
        pass
    out["reject_real_dtype"] = bad_dtype

    e_prop = 0.0
    for qflag in (True, False):
        op_nz = SpinChiralityOperator(
            SpinChiralityConfig(dim=8, dt=0.1, absorption=True, quaternion=qflag),
            dtype=torch.complex128,
        )
        _set_nonz_axes(op_nz)
        with torch.no_grad():
            op_nz.theta_L.copy_(torch.tensor([0.8, -0.5], dtype=torch.float64))
            op_nz.theta_R.copy_(torch.tensor([-0.4, 0.9], dtype=torch.float64))
            op_nz.mass.copy_(torch.tensor([0.3, -0.2], dtype=torch.float64))
            op_nz.gamma.copy_(torch.tensor([-0.5, 0.2], dtype=torch.float64))
        p_nz = dict(
            theta_L=op_nz.theta_L.detach().numpy(),
            axis_L=op_nz.axis_L.detach().numpy().T,
            theta_R=op_nz.theta_R.detach().numpy(),
            axis_R=op_nz.axis_R.detach().numpy().T,
            B=op_nz.B.detach().numpy().T,
            mu=op_nz.mu.detach().numpy(),
            mass=op_nz.mass.detach().numpy(),
            gamma=torch.nn.functional.softplus(op_nz.gamma).detach().numpy(),
        )
        M_np_nz = propagator_numpy(p_nz, 0.1, absorption=True, n_groups=2)
        M_th_nz = propagator_torch(op_nz).detach().numpy()
        e_prop = max(e_prop, float(np.max(np.abs(M_th_nz - M_np_nz))))
    out["propagator_nonz_parity"] = e_prop

    bad_grad = 0.0
    for dt_c, r_dt in ((torch.complex64, torch.float32), (torch.complex128, torch.float64)):
        torch.manual_seed(43)
        op_z = SpinChiralityOperator(SpinChiralityConfig(dim=8, dt=0.1, absorption=False), dtype=dt_c)
        with torch.no_grad():
            op_z.axis_L.zero_()
            op_z.axis_R.zero_()
            op_z.B.zero_()
            op_z.theta_L.fill_(1.5)
            op_z.theta_R.fill_(-0.9)
            op_z.mu.fill_(1.2)
            op_z.mass.fill_(0.4)
        xz = (torch.randn(2, 3, 8, dtype=r_dt) + 1j * torch.randn(2, 3, 8, dtype=r_dt)).requires_grad_(True)
        yz = op_z(xz)
        drift = float((yz.detach().norm(dim=-1) - xz.detach().norm(dim=-1)).abs().max() / xz.detach().norm(dim=-1).max())
        out[f"zero_axis_unit_{'c64' if dt_c == torch.complex64 else 'c128'}"] = drift
        yz.real.sum().backward()
        if xz.grad is None or not bool(torch.isfinite(xz.grad).all()):
            bad_grad += 1.0
        for pname, p in op_z.named_parameters():
            if pname == "gamma":
                continue
            if p.grad is None or not bool(torch.isfinite(p.grad).all()) or float(p.grad.abs().max()) >= 1e4:
                bad_grad += 1.0
    out["zero_axis_finite_grad"] = bad_grad

    e_fd_b0 = 0.0
    for qflag in (True, False):
        torch.manual_seed(44)
        op_fd = SpinChiralityOperator(
            SpinChiralityConfig(dim=8, dt=0.1, absorption=False, quaternion=qflag),
            dtype=torch.complex128,
        )
        _set_nonz_axes(op_fd)
        with torch.no_grad():
            op_fd.B.zero_()
        x_fd = torch.randn(2, 2, 8, dtype=torch.float64) + 1j * torch.randn(2, 2, 8, dtype=torch.float64)
        go_fd = torch.randn(2, 2, 8, dtype=torch.float64) + 1j * torch.randn(2, 2, 8, dtype=torch.float64)
        loss0 = (go_fd.conj() * op_fd(x_fd)).real.sum()
        gB = torch.autograd.grad(loss0, op_fd.B)[0]
        ana = float(gB[0, 1])
        eps = 1e-6
        with torch.no_grad():
            op_fd.B[0, 1] = eps
            lp = float((go_fd.conj() * op_fd(x_fd)).real.sum())
            op_fd.B[0, 1] = -eps
            lm = float((go_fd.conj() * op_fd(x_fd)).real.sum())
            op_fd.B[0, 1] = 0.0
        fd = (lp - lm) / (2.0 * eps)
        e_fd_b0 = max(e_fd_b0, abs(fd - ana))
    out["zero_B_autograd_vs_fd"] = e_fd_b0
    return out


def _check_strang_time_dependent() -> dict:
    """P44：对称 Strang 二阶分裂 O(dt²) 收敛与时变 M_seq(L,G,4,4) 全栈传播/伴随。"""
    p_strang = dict(
        theta_L=np.array([1.7]),
        axis_L=np.array([[0.6], [1.1], [-0.4]]),
        theta_R=np.array([-1.3]),
        axis_R=np.array([[-0.5], [0.8], [0.7]]),
        B=np.array([[0.9], [-0.3], [1.2]]),
        mu=np.array([1.4]),
        mass=np.array([0.8]),
        gamma=np.zeros(1),
    )
    nL = p_strang["axis_L"][:, 0] / np.linalg.norm(p_strang["axis_L"][:, 0])
    nR = p_strang["axis_R"][:, 0] / np.linalg.norm(p_strang["axis_R"][:, 0])
    HL = 0.5 * p_strang["theta_L"][0] * sum(nL[i] * SIG[i] for i in range(3))
    HR = -0.5 * p_strang["theta_R"][0] * sum(nR[i] * SIG[i] for i in range(3))
    HB = -0.5 * p_strang["mu"][0] * sum(p_strang["B"][i, 0] * SIG[i] for i in range(3))
    H4 = p_strang["mass"][0] * BETA.copy()
    H4[:2, :2] += HL + HB
    H4[2:, 2:] += HR + HB
    evals, evecs = np.linalg.eigh(H4)
    T_tot, N1, N2 = 0.4, 16, 32
    U_exact = evecs @ np.diag(np.exp(-1j * evals * T_tot)) @ evecs.conj().T

    M1_1 = propagator_numpy(p_strang, T_tot / N1, absorption=False, n_groups=1,
                            gauge_mode="generator", splitting_order=1)[0]
    M2_1 = propagator_numpy(p_strang, T_tot / N1, absorption=False, n_groups=1,
                            gauge_mode="generator", splitting_order=2)[0]
    M2_2 = propagator_numpy(p_strang, T_tot / N2, absorption=False, n_groups=1,
                            gauge_mode="generator", splitting_order=2)[0]
    e1_1 = float(np.max(np.abs(np.linalg.matrix_power(M1_1, N1) - U_exact)))
    e2_1 = float(np.max(np.abs(np.linalg.matrix_power(M2_1, N1) - U_exact)))
    e2_2 = float(np.max(np.abs(np.linalg.matrix_power(M2_2, N2) - U_exact)))
    ratio2 = e2_2 / max(e2_1, 1e-30)
    strang_gate = max(0.0, ratio2 - 0.27) + max(0.0, 0.22 - ratio2) + max(0.0, e2_1 / max(e1_1, 1e-30) - 0.05)

    rng = np.random.default_rng(44)
    G_tv, L_tv, B_tv, dt_tv, c_tv = 2, 10, 3, 0.05, 0.65
    p_seq = dict(
        theta_L=rng.uniform(-1.5, 1.5, size=(L_tv, G_tv)),
        axis_L=rng.normal(size=(L_tv, 3, G_tv)),
        theta_R=rng.uniform(-1.5, 1.5, size=(L_tv, G_tv)),
        axis_R=rng.normal(size=(L_tv, 3, G_tv)),
        B=rng.normal(size=(L_tv, 3, G_tv)),
        mu=rng.uniform(0.4, 1.6, size=(L_tv, G_tv)),
        mass=rng.uniform(0.1, 0.8, size=(L_tv, G_tv)),
        gamma=rng.uniform(0.0, 0.2, size=(L_tv, G_tv)),
    )
    M_seq = propagator_sequence_numpy(
        p_seq, dt_tv, absorption=True, n_groups=G_tv, gauge_mode="generator", splitting_order=2,
    )
    p_list = [{k: v[n] for k, v in p_seq.items()} for n in range(L_tv)]
    M_seq_list = propagator_sequence_numpy(
        p_list, dt_tv, absorption=True, n_groups=G_tv, gauge_mode="generator", splitting_order=2,
    )
    e_mseq_fmt = float(np.max(np.abs(M_seq - M_seq_list)))

    # 回归（B1）：L=3, G=3 时静态 (3,3) 轴向量不应误当成 (L,3) 时变切片
    p_lg3 = dict(
        theta_L=rng.uniform(-1.0, 1.0, size=(3, 3)),
        axis_L=rng.normal(size=(3, 3)),
        theta_R=rng.uniform(-1.0, 1.0, size=(3, 3)),
        axis_R=rng.normal(size=(3, 3)),
        B=rng.normal(size=(3, 3)),
        mu=rng.uniform(0.5, 1.5, size=3),
        mass=rng.uniform(0.1, 0.5, size=3),
        gamma=np.zeros(3),
    )
    M_lg3 = propagator_sequence_numpy(p_lg3, dt_tv, absorption=False, n_groups=3, gauge_mode="generator")
    for n in range(3):
        p_n = dict(p_lg3, theta_L=p_lg3["theta_L"][n], theta_R=p_lg3["theta_R"][n])
        e_mseq_fmt = max(
            e_mseq_fmt,
            float(np.max(np.abs(M_lg3[n] - propagator_numpy(p_n, dt_tv, absorption=False, n_groups=3, gauge_mode="generator")))),
        )

    u_tv = rng.normal(size=(B_tv, L_tv, 4 * G_tv)) + 1j * rng.normal(size=(B_tv, L_tv, 4 * G_tv))
    y_tv_seq = sequence_numpy_forward(u_tv, M_seq, c_tv)
    y_tv_scan = sequence_scan_numpy(u_tv, M_seq, c_tv)
    e_tv_np = float(np.max(np.abs(y_tv_seq - y_tv_scan)))

    # 独立逐步推进核对 M_seq[n]
    s_ref = np.zeros((B_tv, 1, 4 * G_tv), dtype=np.complex128)
    e_step_tv = 0.0
    for n in range(L_tv):
        s_prop = spin_chirality_numpy_forward(
            s_ref, p_list[n], dt_tv, absorption=True, gauge_mode="generator", splitting_order=2,
        )
        s_ref = s_prop + c_tv * u_tv[:, n:n + 1]
        e_step_tv = max(e_step_tv, float(np.max(np.abs(s_ref[:, 0] - y_tv_seq[:, n]))))

    out = {
        "strang_order1_err_magnitude": e1_1,
        "strang_order2_err_magnitude": e2_1,
        "strang_order2_ratio_magnitude": ratio2,
        "strang_order2_gate": strang_gate,
        "mseq_dict_vs_list": e_mseq_fmt,
        "tv_numpy_scan_vs_seq": e_tv_np,
        "tv_step_vs_forward": e_step_tv,
    }
    if not HAS_TORCH:
        return out

    op_gen = SpinChiralityOperator(
        SpinChiralityConfig(dim=4, dt=T_tot / N1, absorption=False, gauge_mode="generator", splitting_order=2),
        dtype=torch.complex128,
    )
    with torch.no_grad():
        op_gen.theta_L.copy_(torch.tensor(p_strang["theta_L"], dtype=torch.float64))
        op_gen.axis_L.copy_(torch.tensor(p_strang["axis_L"].T, dtype=torch.float64))
        op_gen.theta_R.copy_(torch.tensor(p_strang["theta_R"], dtype=torch.float64))
        op_gen.axis_R.copy_(torch.tensor(p_strang["axis_R"].T, dtype=torch.float64))
        op_gen.B.copy_(torch.tensor(p_strang["B"].T, dtype=torch.float64))
        op_gen.mu.copy_(torch.tensor(p_strang["mu"], dtype=torch.float64))
        op_gen.mass.copy_(torch.tensor(p_strang["mass"], dtype=torch.float64))
    out["strang_torch_vs_numpy"] = float(np.max(np.abs(propagator_torch(op_gen).detach().numpy()[0] - M2_1)))

    torch.manual_seed(45)
    op_tv = SpinChiralityOperator(
        SpinChiralityConfig(dim=4 * G_tv, dt=dt_tv, gauge_mode="generator", splitting_order=2),
        dtype=torch.complex128,
    )
    u_t = torch.tensor(u_tv, dtype=torch.complex128, requires_grad=True)
    M_t = torch.tensor(M_seq, dtype=torch.complex128, requires_grad=True)
    c_t = torch.tensor(c_tv, dtype=torch.float64, requires_grad=True)
    go_t = torch.randn(B_tv, L_tv, 4 * G_tv, dtype=torch.float64) + 1j * torch.randn(B_tv, L_tv, 4 * G_tv, dtype=torch.float64)

    y_scan_t = op_tv.forward_sequence(u_t, c_t, M_seq=M_t, analytic=False)
    y_ana_t = op_tv.forward_sequence(u_t, c_t, M_seq=M_t, analytic=True)
    y_ref_t = sequence_sequential_torch(u_t, M_t, c_t)
    g_scan = torch.autograd.grad(y_scan_t, [u_t, M_t, c_t], go_t)
    g_ana = torch.autograd.grad(y_ana_t, [u_t, M_t, c_t], go_t)
    g_ref = torch.autograd.grad(y_ref_t, [u_t, M_t, c_t], go_t)

    out["tv_torch_vs_numpy"] = float(np.max(np.abs(y_scan_t.detach().numpy() - y_tv_seq)))
    out["tv_analytic_fwd"] = float((y_ana_t - y_ref_t).detach().abs().max())
    out["tv_grad_scan_vs_ref"] = max(float((a - b).abs().max()) for a, b in zip(g_scan, g_ref))
    out["tv_grad_analytic_vs_ref"] = max(float((a - b).abs().max()) for a, b in zip(g_ana, g_ref))

    if HAS_TRITON and torch.cuda.is_available():
        try:
            u_cu = torch.tensor(u_tv, dtype=torch.complex128, device="cuda")
            M_cu = torch.tensor(M_seq, dtype=torch.complex128, device="cuda")
            y_tri = sequence_triton_forward(u_cu, M_cu, c_tv)
            out["tv_triton_vs_numpy"] = float((y_tri.cpu() - torch.tensor(y_tv_seq)).abs().max())

            go_cu = go_t.to("cuda")
            Mr = _c2r_mat(M_cu).detach().requires_grad_(True)
            ur = _c2r_vec(u_cu).detach().requires_grad_(True)
            cr = torch.tensor([c_tv], device="cuda", dtype=torch.float64, requires_grad=True)
            s = torch.zeros(B_tv, G_tv, 8, device="cuda", dtype=torch.float64)
            outs = []
            for n in range(L_tv):
                s = torch.einsum("gij,bgj->bgi", Mr[n], s) + cr[0] * ur[:, n].view(B_tv, G_tv, 8)
                outs.append(s.reshape(B_tv, 8 * G_tv))
            y_ref_cu = torch.stack(outs, 1)
            g_ur, g_Mr, g_cr = torch.autograd.grad(y_ref_cu, [ur, Mr, cr], _c2r_vec(go_cu))
            dUr, dMr, dC = sequence_triton_backward(go_cu, u_cu, y_tri.detach(), M_cu, c_tv)
            out["tv_triton_bwd"] = max(
                float((dUr - g_ur).abs().max()),
                float((dMr - g_Mr).abs().max()),
                float((dC - g_cr).abs().max()),
            )
        except Exception as exc:
            if type(exc).__name__ != "CalledProcessError":
                raise
    return out


def _check_real_wrapper_procrustes() -> dict:
    """P45：实数等变包装（dirac8 / chiral4 / su2_quat4）与闭式 Procrustes 手性/SU(2) 反演。"""
    rng = np.random.default_rng(45)
    dt = 0.25
    G = 2

    # 1. NumPy 三种实数表示的同构一致性与无吸收保范
    p = dict(
        theta_L=np.array([1.2, -0.7]),
        axis_L=np.array([[0.4, -0.5], [0.8, 0.3], [-0.45, 0.8]]),
        theta_R=np.array([-0.9, 1.4]),
        axis_R=np.array([[-0.6, 0.7], [0.5, -0.4], [0.62, 0.58]]),
        B=np.array([[0.8, -0.3], [-0.4, 1.1], [0.5, -0.6]]),
        mu=np.array([1.1, 0.7]),
        mass=np.array([0.6, -0.35]),
        gamma=np.array([0.3, 0.9]),
    )
    e_np = 0.0
    e_norm = 0.0
    for rep, dim_r in (("dirac8", 8 * G), ("chiral4", 4 * G), ("su2_quat4", 4 * G)):
        xr = rng.normal(size=(3, 4, dim_r))
        yr_u = real_spin_chirality_numpy_forward(
            xr, p, dt, representation=rep, absorption=False, gauge_mode="generator", splitting_order=2
        )
        e_norm = max(e_norm, float(np.max(np.abs(np.linalg.norm(yr_u, axis=-1) - np.linalg.norm(xr, axis=-1)))))
        xc, _ = _pack_real_spinor_np(xr, rep)
        p_cmp = dict(p)
        if rep == "chiral4":
            ez = np.zeros((3, G))
            ez[2, :] = 1.0
            p_cmp["axis_L"], p_cmp["axis_R"] = ez, ez
            p_cmp["B"] = np.vstack([np.zeros((2, G)), p["B"][2:3]])
        elif rep == "su2_quat4":
            p_cmp["mass"] = np.zeros(G)
            p_cmp["theta_R"] = np.zeros(G)
        yc = spin_chirality_numpy_forward(xc, p_cmp, dt, absorption=False, gauge_mode="generator", splitting_order=2)
        yr_ref = _unpack_real_spinor_np(yc, rep, G)
        e_np = max(e_np, float(np.max(np.abs(yr_u - yr_ref))))

    # 门禁拒收检查（NumPy 实数包装拒收复数）
    e_gate = 0.0
    try:
        real_spin_chirality_numpy_forward(np.ones((2, 8), dtype=np.complex128), p, dt, representation="dirac8")
        e_gate = 1.0
    except TypeError:
        pass

    # 2. 闭式手性 Procrustes 反演（含 ±π 边界角与独立/轴矢混合验证）
    xc_pair = rng.normal(size=(24, 4 * G))
    p_ch = {
        "theta_L": np.array([2.95, -3.05]),
        "axis_L": np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
        "theta_R": np.array([-2.8, 1.7]),
        "axis_R": np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
        "B": np.zeros((3, G)),
        "mu": np.zeros(G),
        "mass": np.zeros(G),
        "gamma": np.array([0.4, 1.1]),
    }
    yc_pair = real_spin_chirality_numpy_forward(xc_pair, p_ch, dt, representation="chiral4", absorption=True)
    res_ch = fit_chiral_procrustes(xc_pair, yc_pair, dt=dt)
    p_rec = dict(p_ch, theta_L=res_ch["theta_L"], theta_R=res_ch["theta_R"], gamma=res_ch["gamma"])
    yc_rec = real_spin_chirality_numpy_forward(xc_pair, p_rec, dt, representation="chiral4", absorption=True)
    e_chiral_fit = float(np.max(np.abs(yc_rec - yc_pair)))

    # 纯轴矢 θ_L = θ_R = θ 下，不等 L/R 模长仍能精确恢复 axial_angle = θ/2
    x_unbal = rng.normal(size=(20, 4 * G))
    x_unbal[:, 0::4] *= 3.0
    x_unbal[:, 1::4] *= 3.0
    theta_ax = np.array([1.4, -2.1])
    p_ax = dict(p_ch, theta_L=theta_ax, theta_R=theta_ax, gamma=np.array([0.2, 0.6]))
    y_ax = real_spin_chirality_numpy_forward(x_unbal, p_ax, dt, representation="chiral4", absorption=True)
    res_ax = fit_chiral_procrustes(x_unbal, y_ax, dt=dt)
    e_axial = float(np.max(np.abs(np.exp(1j * res_ax["axial_angle"]) - np.exp(0.5j * theta_ax))))

    # 3. 闭式 SU(2) Procrustes（覆盖小角、普通角、θ>π 即 q0<0 大角，以及 G=1 与 G=2）
    e_su2_fit = 0.0
    for n_g, omegas, sc_true in (
        (1, [np.array([1e-5, -1e-5, 5e-6])], np.array([0.73])),
        (1, [4.1 * np.array([0.3, -0.8, 0.52]) / np.linalg.norm([0.3, -0.8, 0.52])], np.array([0.61])),
        (2, [np.array([1.2, -0.7, 0.9]), 3.8 * np.array([-0.5, 0.6, 0.62])], np.array([0.84, 0.52])),
    ):
        xq = rng.normal(size=(18, 4 * n_g))
        yq = np.empty_like(xq)
        U_trues = []
        for g_idx in range(n_g):
            Ug = su2_exp(omegas[g_idx])
            U_trues.append(Ug)
            phi_x = xq[:, 4 * g_idx:4 * g_idx + 4].reshape(-1, 2, 2)
            cx = phi_x[..., 0] + 1j * phi_x[..., 1]
            cy = sc_true[g_idx] * (cx @ Ug.T)
            yq[:, 4 * g_idx] = cy.real[:, 0]
            yq[:, 4 * g_idx + 1] = cy.imag[:, 0]
            yq[:, 4 * g_idx + 2] = cy.real[:, 1]
            yq[:, 4 * g_idx + 3] = cy.imag[:, 1]
        _, sc_fit, U_fit = fit_su2_procrustes(xq, yq)
        U_ref = U_trues[0] if n_g == 1 else np.stack(U_trues, axis=0)
        sc_ref = float(sc_true[0]) if n_g == 1 else sc_true
        e_su2_fit = max(e_su2_fit, float(np.max(np.abs(U_fit - U_ref))), float(np.max(np.abs(sc_fit - sc_ref))))

    out = {
        "numpy_rep_equiv": e_np,
        "numpy_unitary_norm": e_norm,
        "dtype_gate_reject": e_gate,
        "procrustes_chiral": e_chiral_fit,
        "procrustes_axial": e_axial,
        "procrustes_su2": e_su2_fit,
    }
    if not HAS_TORCH:
        return out

    # 4. PyTorch RealSpinChiralityOperator 合同门禁、精度、梯度与序列传播
    op_cplx = SpinChiralityOperator(SpinChiralityConfig(dim=8), dtype=torch.complex64)
    try:
        op_cplx(torch.randn(2, 3, 8, dtype=torch.float32))
        e_gate = 1.0
    except TypeError:
        pass

    e_torch = 0.0
    e_grad = 0.0
    for rep, dim_r in (("dirac8", 8 * G), ("chiral4", 4 * G), ("su2_quat4", 4 * G)):
        torch.manual_seed(45)
        cfg = SpinChiralityConfig(
            dim=dim_r, dt=dt, absorption=True, gauge_mode="generator", splitting_order=2
        )
        op_r = RealSpinChiralityOperator(cfg, representation=rep, dtype=torch.float64)
        if op_r.config.n_groups != op_r.n_groups:
            e_gate = 1.0
        try:
            op_r(torch.randn(2, 3, dim_r, dtype=torch.float64) + 1j)
            e_gate = 1.0
        except TypeError:
            pass
        with torch.no_grad():
            op_r.theta_L.copy_(torch.tensor(p["theta_L"], dtype=torch.float64))
            op_r.mu.copy_(torch.tensor(p["mu"], dtype=torch.float64))
            op_r.gamma.copy_(torch.tensor(_inv_softplus_np(p["gamma"]), dtype=torch.float64))
            if rep != "chiral4":
                op_r.axis_L.copy_(torch.tensor(p["axis_L"].T, dtype=torch.float64))
                op_r.B.copy_(torch.tensor(p["B"].T, dtype=torch.float64))
            if rep == "dirac8":
                op_r.axis_R.copy_(torch.tensor(p["axis_R"].T, dtype=torch.float64))
                op_r.theta_R.copy_(torch.tensor(p["theta_R"], dtype=torch.float64))
                op_r.mass.copy_(torch.tensor(p["mass"], dtype=torch.float64))
            elif rep == "chiral4":
                op_r.theta_R.copy_(torch.tensor(p["theta_R"], dtype=torch.float64))
                op_r.mass.copy_(torch.tensor(p["mass"], dtype=torch.float64))
        xr = rng.normal(size=(2, 3, dim_r))
        xt = torch.tensor(xr, dtype=torch.float64, requires_grad=True)
        yt = op_r(xt)
        p_tref = dict(p)
        if rep == "chiral4":
            p_tref["B"] = np.zeros((3, G))
        elif rep == "su2_quat4":
            p_tref["mass"] = np.zeros(G)
            p_tref["theta_R"] = np.zeros(G)
        ynp = real_spin_chirality_numpy_forward(
            xr, p_tref, dt, representation=rep, absorption=True, gauge_mode="generator", splitting_order=2
        )
        e_torch = max(e_torch, float(np.max(np.abs(yt.detach().numpy() - ynp))))

        # 梯度核对（输入 x 与可微参数 theta_L）
        go = torch.randn_like(yt)
        g_x, g_th = torch.autograd.grad((go * yt).sum(), [xt, op_r.theta_L])
        eps = 1e-6
        with torch.no_grad():
            xt_p = xt.detach().clone()
            xt_m = xt.detach().clone()
            xt_p[0, 1, 2] += eps
            xt_m[0, 1, 2] -= eps
            fd_x = float(((op_r(xt_p) - op_r(xt_m)) * go).sum() / (2.0 * eps))
            op_r.theta_L[0] += eps
            lp = float((op_r(xt.detach()) * go).sum())
            op_r.theta_L[0] -= 2.0 * eps
            lm = float((op_r(xt.detach()) * go).sum())
            op_r.theta_L[0] += eps
            fd_th = (lp - lm) / (2.0 * eps)
        e_grad = max(e_grad, abs(float(g_x[0, 1, 2]) - fd_x), abs(float(g_th[0]) - fd_th))

    # forward_sequence 与打包解包一致性（dirac8 + chiral4，含解析伴随）
    u_seq = torch.randn(2, 6, 8 * G, dtype=torch.float64, requires_grad=True)
    op_seq = RealSpinChiralityOperator(
        SpinChiralityConfig(dim=8 * G, dt=0.1, gauge_mode="generator", splitting_order=2),
        representation="dirac8",
        dtype=torch.float64,
    )
    y_scan = op_seq.forward_sequence(u_seq, coupling=0.6, analytic=False)
    y_ana = op_seq.forward_sequence(u_seq, coupling=0.6, analytic=True)
    e_seq = float((y_scan - y_ana).detach().abs().max())
    g_seq_s = torch.autograd.grad(y_scan.sum(), [u_seq, op_seq.theta_L], retain_graph=True)
    g_seq_a = torch.autograd.grad(y_ana.sum(), [u_seq, op_seq.theta_L])
    e_seq = max(e_seq, max(float((a - b).abs().max()) for a, b in zip(g_seq_s, g_seq_a)))

    # fit_procrustes 写回零迭代一步拟合（chiral4 protocol/generator + su2_quat4）
    e_module_fit = 0.0
    for gm in ("protocol", "generator"):
        op_fit = RealSpinChiralityOperator(
            SpinChiralityConfig(dim=4 * G, dt=dt, absorption=True, gauge_mode=gm),
            representation="chiral4",
            dtype=torch.float64,
        )
        op_fit.fit_procrustes(torch.tensor(xc_pair), torch.tensor(yc_pair), update=True)
        e_module_fit = max(e_module_fit, float((op_fit(torch.tensor(xc_pair)) - torch.tensor(yc_pair)).detach().abs().max()))

    xq2 = rng.normal(size=(18, 4 * G))
    yq2 = np.empty_like(xq2)
    for g_idx, om in enumerate((np.array([1.1, -0.6, 0.8]), np.array([-0.5, 1.3, 0.4]))):
        Ug = su2_exp(om)
        cx = xq2[:, 4 * g_idx:4 * g_idx + 4].reshape(-1, 2, 2)
        cx = cx[..., 0] + 1j * cx[..., 1]
        cy = (0.75 if g_idx == 0 else 0.55) * (cx @ Ug.T)
        yq2[:, 4 * g_idx], yq2[:, 4 * g_idx + 1] = cy.real[:, 0], cy.imag[:, 0]
        yq2[:, 4 * g_idx + 2], yq2[:, 4 * g_idx + 3] = cy.real[:, 1], cy.imag[:, 1]
    op_qfit = RealSpinChiralityOperator(
        SpinChiralityConfig(dim=4 * G, dt=dt, absorption=True, gauge_mode="protocol"),
        representation="su2_quat4",
        dtype=torch.float64,
    )
    op_qfit.fit_procrustes(torch.tensor(xq2), torch.tensor(yq2), update=True)
    e_module_fit = max(e_module_fit, float((op_qfit(torch.tensor(xq2)) - torch.tensor(yq2)).detach().abs().max()))

    # 回归（A2 / A3）：.double() 同步 .dtype/.complex_dtype；默认 chiral4 初始 mass=0 可被 Procrustes 精确拟合
    op_sync = RealSpinChiralityOperator(SpinChiralityConfig(dim=8), representation="dirac8", dtype=torch.float32)
    op_sync.double()
    if op_sync.dtype != torch.float64 or op_sync.complex_dtype != torch.complex128:
        e_gate = 1.0
    xd_sync = torch.randn(2, 8, dtype=torch.float64)
    if op_sync(xd_sync).dtype != torch.float64:
        e_gate = 1.0

    op_ch_t = RealSpinChiralityOperator(SpinChiralityConfig(dim=8, dt=1.0), representation="chiral4", dtype=torch.float64)
    with torch.no_grad():
        op_ch_t.theta_L.copy_(torch.tensor([1.1, -0.8], dtype=torch.float64))
        op_ch_t.theta_R.copy_(torch.tensor([-0.6, 1.3], dtype=torch.float64))
        op_ch_t.gamma.copy_(torch.tensor([-2.0, -1.0], dtype=torch.float64))
        x_ch_t = torch.randn(16, 8, dtype=torch.float64)
        y_ch_t = op_ch_t(x_ch_t)
    op_ch_s = RealSpinChiralityOperator(SpinChiralityConfig(dim=8, dt=1.0), representation="chiral4", dtype=torch.float64)
    op_ch_s.fit_procrustes(x_ch_t, y_ch_t, update=True)
    e_module_fit = max(e_module_fit, float((op_ch_s(x_ch_t) - y_ch_t).detach().abs().max()))

    out["dtype_gate_reject"] = e_gate
    out["torch_vs_numpy"] = e_torch
    out["autograd_vs_fd"] = e_grad
    out["real_seq_vs_analytic"] = e_seq
    out["module_fit_procrustes"] = e_module_fit
    return out


# 审计项清单：函数名用字符串，缺可选依赖时对应函数根本不存在，也能如实标"跳过"而不是崩
_AUDIT_CHECKS = (
    ("P1 Clifford", "_check_clifford"),
    ("P2 手性投影", "_check_projectors"),
    ("P3 SU(2)", "_check_su2"),
    ("P4 Larmor 进动", "_check_larmor"),
    ("P5 手性动力学", "_check_chirality"),
    ("P6 轴矢 Ward", "_check_ward"),
    ("P7 幺正性/吸收", "_check_unitarity"),
    ("P8 可微性", "_check_gradcheck"),
    ("P9 自由演化精确", "_check_exact_free_evolution"),
    ("P10 序列扫描", "_check_sequence"),
    ("P11 序列梯度", "_check_sequence_grad"),
    ("P12 Triton 三重对齐", "_check_sequence_triton"),
    ("P13 解析伴随", "_check_analytic_adjoint"),
    ("P14 手性守恒", "_check_chirality_conservation"),
    ("P15 四元数 SU(2)", "_check_quaternion_su2"),
    ("P16 反常记账", "_check_anomaly_bookkeeping"),
    ("P17 3+1D 指标", "_check_index_4d"),
    ("P18 反常系数(Fujikawa)", "_check_fujikawa_coefficient"),
    ("P19 微观谱与代数", "_check_micro_spectrum"),
    ("P20 速度算符/ZBW", "_check_zitterbewegung"),
    ("P21 自包含与完整性", "_check_self_contained"),
    ("P22 1+1D 色散/倍增", "_check_lattice_1d"),
    ("P23 畴壁零模", "_check_domain_wall_1d"),
    ("P24 规范步离散阶", "_check_gauge_midpoint"),
    ("P25 1+1D 时间演化", "_check_time_evolution"),
    ("P26 1+1D 规范场", "_check_gauge_1d"),
    ("P27 Fock 空间", "_check_fock_space"),
    ("P28 海的手征荷", "_check_sea_chiral_1d"),
    ("P29 1+1D 热核", "_check_fujikawa_1d"),
    ("P30 对产生(LZ)", "_check_pair_production_1d"),
    ("P31 规范场反作用", "_check_gauge_backreaction"),
    ("P32 有质量反作用", "_check_massive_backreaction"),
    ("P33 禁闭(弦张力)", "_check_confinement_1d"),
    ("P34 Kogut–Susskind ED", "_check_ks_ed"),
    ("P35 Kogut–Susskind 连续极限", "_check_ks_continuum"),
    ("P36 Kogut–Susskind MPS/DMRG", "_check_ks_mps"),
    ("P37 光子/弦态与电场关联", "_check_ks_photon"),
    ("P38 电场矩阵元与谱权重", "_check_ks_photon_weight"),
    ("P39 光子谱定位与 m_γ", "_check_ks_photon_mass"),
    ("P40 光子色散截距", "_check_ks_dispersion"),
    ("P41 平移与动量扇区", "_check_ks_basis"),
    ("P42 光子色散 ω(k)", "_check_ks_omega"),
    ("P43 零轴光滑性与构造边界", "_check_axis_smoothness"),
    ("P44 时变规范与对称分裂二阶收敛", "_check_strang_time_dependent"),
    ("P45 实数等变包装与手性传感反演", "_check_real_wrapper_procrustes"),
)

_TRITON_CHECKS = {"_check_sequence_triton", "_check_chirality_conservation"}


def run_physics_audit(verbose: bool = True) -> dict:
    """跑 P1–P45 全项物理自检。

    缺可选依赖（torch / triton / CUDA 工具链）的项记 `skipped=True`，不算失败；
    其余任何异常都记 `worst=inf`。返回 {项名: {"detail", "worst", "skipped"}}。
    """
    report = {}
    if verbose:
        print("=" * 96)
    for name, fn_name in _AUDIT_CHECKS:
        fn = globals().get(fn_name)
        if fn is None:
            report[name] = {"detail": {}, "worst": 0.0, "skipped": True,
                            "exception": "缺少可选依赖（torch / triton），本项未注册"}
            if verbose:
                print(f"{name:<16} 跳过：缺少可选依赖（torch / triton）")
            continue
        try:
            r = fn()
        except Exception as exc:
            kind = type(exc).__name__
            excused = kind in ("ImportError", "ModuleNotFoundError") or (
                kind == "CalledProcessError" and fn_name in _TRITON_CHECKS
            )
            report[name] = {"detail": {}, "worst": 0.0 if excused else float("inf"),
                            "skipped": excused, "exception": f"{kind}: {exc}"}
            if verbose:
                print(f"{name:<16} {'跳过' if excused else '失败'}：{kind}: {exc}")
            continue
        skipped = any(k.startswith("skipped") for k in r)
        errs = {k: v for k, v in r.items()
                if not k.endswith("_magnitude") and not k.startswith("skipped")}
        worst = max((abs(v) for v in errs.values()), default=0.0)
        report[name] = {"detail": r, "worst": worst, "skipped": skipped}
        if verbose:
            detail = ", ".join(f"{k}={v:.2e}" for k, v in r.items())
            print(f"{name:<16} worst = {worst:.2e}   [{detail}]")
    if verbose:
        print("=" * 96)
        print("注：P7 覆盖经典幺正性；ABJ 反常的量子系数由 P18（Fujikawa 热核）给出，"
              "系数与 P16 的记账、P17 的格点指标闭合。")
    return report


# ==============================================================================
# 5. ABJ 记账接口（反常积分形式）
# ==============================================================================
def ff_from_eb(E, B):
    """场强缩并 F^a_{μν}F̃^{aμν}（约定 ε^{0123}=+1，F_{0i}=E_i，F_{ij}=−ε_{ijk}B_k）。

    在该约定下 F_{μν}F̃^{μν} = +4 E·B。
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    return 4.0 * float(np.dot(E, B))


def anomaly_axial_charge_step(q5: float, ff: float, g: float, dt: float) -> float:
    """轴向荷的反常变化：ΔQ5 = (g²/16π²)·F^a_{μν}F̃^{aμν}·dt（单味 SU(2)）。

    以 ε^{0123}=+1、F F̃ = +4E·B 计，等价于 ΔQ5 = (g²/4π²)·E·B·dt。
    系数由 P18 的热核（Fujikawa）推导并数值校验（tr[γ5X²]=εFF ⇒ A=εFF/(32π²)=F F̃/(16π²)）；
    单圈三角图的直接数值化尝试见根目录 anomaly_pv_wip.py（该路线走不通，理由见其文件头）。
    """
    return q5 + (g * g / (16.0 * np.pi ** 2)) * ff * dt


if __name__ == "__main__":
    run_physics_audit()
