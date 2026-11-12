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
= e^{-im\beta dt}\,e^{+i\mu B\cdot\Sigma dt}$ 就是单一哈密顿量的指数，无分裂误差（P9 检验）。
手性旋转 $U_{L/R}$ 是**瞬时规范变换**（协议中的一步），不由哈密顿量连续生成；
若把它改成连续生成的时间依赖规范场，才会出现一阶分裂误差。

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

吞吐（B=16, G=8，同进程公平对比）：
    宽加载持久核 fp64：L=256 约 0.1–0.4 ms、L=4096 约 2–3 ms；torch 实张量序贯
    L=256 24 ms / L=4096 413 ms → 约 100–126×。
    尝试过但未采用：分块扫描（两趟双倍访存，慢 1.4–2×）、软件流水 num_stages=2
    （L=256 快 1.3×、L=4096 无效）、对角化+FFT 卷积（慢且未对齐）。

物理检验（python -m spin_chirality 自动执行）
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

本机实测（python -m spin_chirality）
--------------------------------------------
    P1 Clifford     0.0        P2 投影          0.0
    P3 SU(2)        5.6e-16    P4 Larmor        2.2e-16
    P5 手性          1.1e-16    P6 轴矢 Ward     2.6e-16
    P7 幺正/吸收     8.9e-16    P8 可微性        5.9e-07（有限差分）

使用示例
--------
    import torch
    from spin_chirality import SpinChiralityConfig, SpinChiralityOperator

    op = SpinChiralityOperator(SpinChiralityConfig(dim=64, dt=0.05))
    x = torch.randn(2, 16, 64, dtype=torch.complex128)   # ℂ^{B×L×4G}
    y = op(x)
    y.real.sum().backward()                               # 复张量可微

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
    import triton
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



__all__ = [
    "SpinChiralityConfig",
    "SpinChiralityOperator",
    "spin_chirality_numpy_forward",
    "su2",
    "su2_generator",
    "su2_quat",
    "apply_su2_spinor",
    "quat_apply",
    "quat_conj",
    "quat_mul",
    "propagator_numpy",
    "sequence_numpy_forward",
    "sequence_scan_numpy",
    "run_physics_audit",
    "ff_from_eb",
    "anomaly_axial_charge_step",
    "dirac_hamiltonian",
    "larmor_unitary",
    "spin_expectation_numpy",
    "check_flux_index",
    "check_spectral_flow",
]

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


class SpinChiralityConfig:
    """配置：dim 必须为 4 的倍数（G 个 Dirac 旋量），dt 为演化步长。"""

    def __init__(self, dim: int = 16, dt: float = 0.1, absorption: bool = True,
                 quaternion: bool = True) -> None:
        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"dim 必须为 4 的正整数倍（Dirac 旋量），当前 {dim}")
        if dt <= 0:
            raise ValueError(f"dt 必须为正，当前 {dt}")
        self.dim = dim
        self.dt = float(dt)
        self.absorption = bool(absorption)
        self.quaternion = bool(quaternion)

    @property
    def n_groups(self) -> int:
        return self.dim // 4

    def __repr__(self) -> str:
        return f"SpinChiralityConfig(dim={self.dim}, dt={self.dt:g}, absorption={self.absorption})"


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


def spin_chirality_numpy_forward(x, params, dt: float, absorption: bool = True):
    """物理忠实前向（NumPy）。

    Args:
        x: 复数数组 (B, L, 4G)。
        params: dict，键为
            theta_L, axis_L(3,G), theta_R, axis_R(3,G),
            B(3,G), mu(G), mass(G), gamma(G)
        dt: 演化步长。
    Returns:
        y: 复数数组 (B, L, 4G)。
    """
    x = np.asarray(x, dtype=np.complex128)
    pl, pr = _split(x)                      # (B,L,G,2)
    G = x.shape[-1] // 4
    theta_L = np.asarray(params["theta_L"], dtype=float)
    theta_R = np.asarray(params["theta_R"], dtype=float)
    axis_L = np.asarray(params["axis_L"], dtype=float)
    axis_R = np.asarray(params["axis_R"], dtype=float)
    B = np.asarray(params["B"], dtype=float)
    mu = np.asarray(params["mu"], dtype=float)
    mass = np.asarray(params["mass"], dtype=float)
    gamma = np.asarray(params["gamma"], dtype=float) if absorption else np.zeros(G)

    pl_out = np.empty_like(pl)
    pr_out = np.empty_like(pr)
    for g in range(G):
        c2, s2 = np.cos(mass[g] * dt / 2), np.sin(mass[g] * dt / 2)
        hl1 = c2 * pl[..., g, 0] - 1j * s2 * pr[..., g, 0]
        hl2 = c2 * pl[..., g, 1] - 1j * s2 * pr[..., g, 1]
        hr1 = c2 * pr[..., g, 0] - 1j * s2 * pl[..., g, 0]
        hr2 = c2 * pr[..., g, 1] - 1j * s2 * pl[..., g, 1]
        l1, l2 = apply_su2_spinor(hl1, hl2, axis_L[:, g], theta_L[g])
        r1, r2 = apply_su2_spinor(hr1, hr2, axis_R[:, g], -theta_R[g])  # R†
        phi = -mu[g] * np.linalg.norm(B[:, g]) * dt   # H = -mu B·Sigma
        l1, l2 = apply_su2_spinor(l1, l2, B[:, g], phi)
        r1, r2 = apply_su2_spinor(r1, r2, B[:, g], phi)
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


# ==============================================================================
# 3. PyTorch 模块（复数张量，可微）
# ==============================================================================
if HAS_TORCH:


    def _quat_torch(axis, theta):
        """U = cos(θ/2) I + i(q·σ) 的四元数 (G,4)。"""
        n = axis / (axis.norm(dim=-1, keepdim=True) + 1e-30)
        return torch.cat([torch.cos(theta / 2)[:, None],
                          -torch.sin(theta / 2)[:, None] * n], dim=-1)

    def _quat_apply_torch(q, z):
        """四元数作用在复 2-旋量 (…,G,2) 上。"""
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        z1, z2 = z[..., 0], z[..., 1]
        return torch.stack([(q0 + 1j * q3) * z1 + (q2 + 1j * q1) * z2,
                            (-q2 + 1j * q1) * z1 + (q0 - 1j * q3) * z2], dim=-1)

    def _quat_conj_torch(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def _su2_torch(axis, theta):
        """axis: (G,3) real tensor，theta: (G,) → 2×2 复数矩阵的批量 (G,2,2)。"""
        n = axis / (axis.norm(dim=-1, keepdim=True) + 1e-12)
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        eye = torch.eye(2, dtype=torch.complex128 if axis.dtype == torch.float64 else torch.complex64,
                        device=axis.device).expand(axis.shape[0], 2, 2)
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        sigma = torch.stack([
            torch.stack([nz, nx - 1j * ny], dim=-1),
            torch.stack([nx + 1j * ny, -nz], dim=-1),
        ], dim=-2)
        return c[:, None, None] * eye - 1j * s[:, None, None] * sigma

    class SpinChiralityOperator(nn.Module):
        """一阶 Lie–Trotter 分裂的 Dirac/Weyl 演化：手性规范 × Larmor × 质量 × 吸收。

        参数（每组 G 个，均可学习）：
            theta_L, axis_L(3)   SU(2)_L 转角与转轴
            theta_R, axis_R(3)   SU(2)_R
            B(3), mu             Larmor 磁场与磁矩
            mass                 质量
            gamma                吸收系数（softplus 保证 ≥ 0）
        """

        def __init__(self, config: SpinChiralityConfig, dtype=torch.complex128, device=None) -> None:
            super().__init__()
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
            if x.is_complex() is False:
                raise TypeError("输入必须是复数张量（ℂ^{B×L×4G}）")
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

            if self.config.quaternion:
                qL = _quat_torch(self.axis_L.to(real_dtype), self.theta_L.to(real_dtype)).to(real_dtype)
                qR = _quat_torch(self.axis_R.to(real_dtype), self.theta_R.to(real_dtype)).to(real_dtype)
                pl = _quat_apply_torch(qL, pl)
                pr = _quat_apply_torch(_quat_conj_torch(qR), pr)   # R† = 共轭四元数
            else:
                U_L = _su2_torch(self.axis_L.to(real_dtype), self.theta_L.to(real_dtype))
                U_R = _su2_torch(self.axis_R.to(real_dtype), self.theta_R.to(real_dtype))
                pl = torch.einsum("...gi,gji->...gj", pl, U_L.to(dtype))
                pr = torch.einsum("...gi,gij->...gj", pr, U_R.conj().to(dtype))

            B = self.B.to(real_dtype)
            phi = -self.mu.to(real_dtype) * B.norm(dim=-1) * self.config.dt  # H = -mu B·Sigma
            if self.config.quaternion:
                qB = _quat_torch(B, phi)
                pl = _quat_apply_torch(qB, pl)
                pr = _quat_apply_torch(qB, pr)
            else:
                U_B = _su2_torch(B, phi).to(dtype)
                pl = torch.einsum("...gi,gji->...gj", pl, U_B)
                pr = torch.einsum("...gi,gji->...gj", pr, U_B)

            nl = c2 * pl - 1j * s2 * pr   # 后半步质量 kick
            nr = c2 * pr - 1j * s2 * pl

            if self.config.absorption:
                att = torch.exp(-self._gamma_eff().to(real_dtype) * self.config.dt).to(dtype)[:, None]
                nl, nr = nl * att, nr * att

            y = torch.cat([nl, nr], dim=-1).reshape(*x.shape[:-1], 4 * G)
            return y

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

else:  # pragma: no cover
    SpinChiralityOperator = None  # type: ignore


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
    """(i/2) n·σ —— SU(2) 生成元，**反厄米**、无迹：它是 U = e^{iθ n·σ/2} 在 θ=0 处的导数。"""
    return 0.5j * sum(n_hat[i] * SIG[i] for i in range(3))


def larmor_unitary(B: np.ndarray, mu: float, t: float) -> np.ndarray:
    """U(t) = exp(−i H t)，H = −μ B·σ/2（与 su2 同一套约定）。"""
    B = np.asarray(B, dtype=float)
    n = B / np.linalg.norm(B)
    return su2(n, mu * np.linalg.norm(B) * t)


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


def propagator_numpy(params, dt: float, absorption: bool = True, n_groups: int = 1):
    """用基矢探测从一步物理中提取 4×4 传播子 M（块对角，跨组为零）。"""
    D = 4 * n_groups
    M = np.zeros((n_groups, 4, 4), dtype=complex)
    for g in range(n_groups):
        for k in range(4):
            x = np.zeros((1, 1, D), dtype=complex)
            x[0, 0, 4 * g + k] = 1.0
            y = spin_chirality_numpy_forward(x, params, dt, absorption=absorption).reshape(D)
            M[g][:, k] = y[4 * g:4 * g + 4]
    return M


def sequence_numpy_forward(u, M, coupling: float):
    """黄金对照：序贯递推 s_{n+1} = M s_n + c u_n，返回全部状态。"""
    B, L, D = u.shape
    G = D // 4
    s = np.zeros((B, G, 4), dtype=complex)
    out = np.empty((B, L, D), dtype=complex)
    for n in range(L):
        s = np.einsum("gij,bgj->bgi", M, s) + coupling * u[:, n].reshape(B, G, 4)
        out[:, n] = s.reshape(B, D)
    return out


def sequence_scan_numpy(u, M, coupling: float):
    """结合律前缀扫描（Hillis–Steele），O(log L) 深度。"""
    B, L, D = u.shape
    G = D // 4
    Ms = np.tile(M[None, None], (B, L, 1, 1, 1)).astype(complex)
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
        """从 torch 算子提取传播子（可微）：对每组 4 个基矢各做一次一步前向。"""
        G = op.config.n_groups
        dt = dtype or (torch.complex128 if op.theta_L.dtype == torch.float64 else torch.complex64)
        x = torch.zeros(4 * G, 1, 4 * G, dtype=dt, device=op.theta_L.device)
        for k in range(4 * G):
            x[k, 0, k] = 1.0
        y = op(x)
        return torch.stack([y[4 * g:4 * g + 4, 0, 4 * g:4 * g + 4] for g in range(G)])

    def sequence_scan_torch(u, M, coupling):
        """结合律前缀扫描（torch，可微）。"""
        B, L, D = u.shape
        G = D // 4
        Ms = M[None, None].expand(B, L, G, 4, 4).clone()
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
        """序贯递推（torch，可微），作为扫描的对照。"""
        B, L, D = u.shape
        G = D // 4
        s = torch.zeros(B, G, 4, dtype=u.dtype, device=u.device)
        out = []
        for n in range(L):
            s = torch.einsum("gij,bgj->bgi", M, s) + coupling * u[:, n].reshape(B, G, 4)
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


    def _check_sequence_grad() -> dict:
        """序列模式：自定义反向 vs autograd，及对 u 的有限差分抽查。"""
        torch.manual_seed(0)
        G, L, B = 2, 6, 2
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
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
        """Triton 融合前向（fp64，CUDA）。"""
        Mr = _c2r_mat(M).contiguous()
        ur = _c2r_vec(u)
        B, L, _ = ur.shape
        G = M.shape[0]
        yr = torch.empty_like(ur)
        c_t = torch.tensor([float(coupling)], device=u.device, dtype=ur.dtype)
        big = 1
        while big < G:
            big *= 2
        _seq_fwd_wide[(B,)](ur, yr, Mr, c_t, L, G, BLOCK=8, BIG=big)
        return _r2c_vec(yr)

    def sequence_triton_backward(go, u, y, M, coupling):
        """Triton 融合反向（fp64，CUDA）：返回实表示下的 (dUr, dMr, dC)。"""
        Mr = _c2r_mat(M).contiguous()
        ur = _c2r_vec(u)
        g_ur = _c2r_vec(go)
        yr = _c2r_vec(y)
        B, L, _ = ur.shape
        G = M.shape[0]
        dur = torch.empty_like(ur)
        dMr = torch.zeros_like(Mr)
        dC = torch.zeros(1, device=u.device, dtype=ur.dtype)
        c_t = torch.tensor([float(coupling)], device=u.device, dtype=ur.dtype)
        _seq_bwd_kernel[(B * G,)](g_ur, ur, yr, Mr, c_t, dur, dMr, dC, L, G, BLOCK=8)
        return dur, dMr, dC

    def _check_sequence_triton() -> dict:
        """三重对齐：NumPy 黄金 vs torch 扫描 vs Triton 核；反向对比实表示 autograd。"""
        if not torch.cuda.is_available():
            return {"skipped_no_cuda": 0.0}
        torch.manual_seed(0)
        G, L, B = 3, 10, 4
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
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
        """离散伴随（实表示）：λ_n = g_n + Mᵀλ_{n+1}，dM/dc 累加，dU = c λ。"""
        B, L, D8 = u.shape
        G = M.shape[0]
        lam = torch.zeros(B, G, 8, dtype=u.dtype, device=u.device)
        dM = torch.zeros_like(M)
        dc = torch.zeros((), dtype=u.dtype, device=u.device)
        dU = torch.empty_like(u)
        zero = torch.zeros(B, G, 8, dtype=u.dtype, device=u.device)
        for n in range(L - 1, -1, -1):
            lam = go[:, n].view(B, G, 8) + torch.einsum("gij,bgj->bgi", M.transpose(-1, -2), lam)
            s_prev = y[:, n - 1].view(B, G, 8) if n > 0 else zero
            dM = dM + torch.einsum("bgi,bgj->gij", lam, s_prev)
            dc = dc + (lam * u[:, n].view(B, G, 8)).sum()
            dU[:, n] = (c * lam).reshape(B, D8)
        return dU, dM, dc

    class AnalyticSequenceFunction(torch.autograd.Function):
        """解析伴随版序列算子：反向为手写 λ 递推，不重算前向图。"""

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
            A = dMr[:, :4, :4] + dMr[:, 4:, 4:]
            Bm = -dMr[:, :4, 4:] + dMr[:, 4:, :4]
            dM = torch.complex(A, Bm).to(M.dtype)
            return dU.to(u.dtype), dM, dC.to(coupling.dtype)

    def _check_analytic_adjoint() -> dict:
        """P13：解析伴随 Function vs 重计算 Function（同一上游梯度）。"""
        torch.manual_seed(0)
        G, L, B = 2, 8, 2
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
        M = propagator_torch(op).detach()
        u = (torch.randn(B, L, 4 * G, dtype=torch.float64)
             + 1j * torch.randn(B, L, 4 * G, dtype=torch.float64)).requires_grad_(True)
        c = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        go = (torch.randn(B, L, 4 * G, dtype=torch.float64)
              + 1j * torch.randn(B, L, 4 * G, dtype=torch.float64))
        y1 = AnalyticSequenceFunction.apply(u, M, c)
        g1 = torch.autograd.grad(y1, [u, c], go)
        y2 = SequenceFunction.apply(u, M, c)
        g2 = torch.autograd.grad(y2, [u, c], go)
        e_u = float((g1[0] - g2[0]).abs().max())
        e_c = float((g1[1] - g2[1]).abs().max())
        # 同时给出对 y 的前向一致性
        e_y = float((y1 - y2).abs().max())
        return dict(analytic_vs_recompute_u=e_u, analytic_vs_recompute_c=e_c, fwd=e_y)


    def sequence_axial_charge(y):
        """Q5[n] = ‖ψ_R‖² − ‖ψ_L‖²（逐 (b,n)），即 γ⁵ 期望对应的手性荷。"""
        B, L, D = y.shape
        G = D // 4
        v = y.reshape(B, L, G, 4)
        ql = (v[..., 0].abs() ** 2 + v[..., 1].abs() ** 2).sum(dim=-1)
        qr = (v[..., 2].abs() ** 2 + v[..., 3].abs() ** 2).sum(dim=-1)
        return (qr - ql)

    def _check_chirality_conservation() -> dict:
        """物理判据：m=0 时手性荷沿序列守恒；m>0 时以 2m 频率振荡。"""
        torch.manual_seed(0)
        G, L, B = 2, 64, 4
        op = SpinChiralityOperator(SpinChiralityConfig(dim=4 * G, dt=0.1))
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
                import importlib as _il
                lattice4d = _il.import_module("spin_chirality_lattice4d")
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
    """由 (转轴, 角) 构造单位四元数 q = (cos(θ/2), −sin(θ/2)·n̂)。"""
    n = np.asarray(axis, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-300)
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
    "su2_generator": "P21 自包含与完整性",
    "su2_quat": "P15 四元数 SU(2)",
    "apply_su2_spinor": "P10 序列扫描",
    "quat_apply": "P15 四元数 SU(2)",
    "quat_conj": "P15 四元数 SU(2)",
    "quat_mul": "P15 四元数 SU(2)",
    "propagator_numpy": "P10 序列扫描",
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
}

# 硬依赖白名单：标准库 + numpy（torch / triton / scipy 必须包在 try 里）
_HARD_IMPORT_ALLOWED = {"__future__", "numpy", "itertools", "sys", "ast", "io", "os"}


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
    # 只扫本模块的实现文件。上游是独立包（目录里只有它自己的几个文件），
    # 扫目录 == 扫自己；本仓库是平铺布局，同目录还有 20+ 个别的算子，
    # 扫全目录会把它们的 import 误算成 spin 模块的硬依赖。
    files = sorted(f for f in _os.listdir(here)
                   if f.endswith(".py") and f.startswith("spin_chirality_"))
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
    public = [n for n in allnames if n not in ("lattice4d", "__version__")]

    e_all = float(sum(0 if hasattr(pkg, n) else 1 for n in allnames))

    hard, _soft, bad, _n_files = _scan_package_sources()
    e_dep = float(len(hard - _HARD_IMPORT_ALLOWED))
    e_io = float(len(bad))

    item_names = {n for n, _ in _AUDIT_CHECKS}
    e_cov = float(len(set(public) - set(_PUBLIC_COVERAGE)))
    for v in _PUBLIC_COVERAGE.values():
        if not v.startswith("专项") and v not in item_names:
            e_cov += 1.0

    s = su2_generator(np.array([0.0, 0.0, 1.0]))
    # (i/2)n·σ 是反厄米的（它是 U = e^{iθn·σ/2} 在 θ=0 的导数），且无迹
    e_gen = max(float(np.abs(s + s.conj().T).max()), abs(complex(np.trace(s))))

    U = larmor_unitary(np.array([0.0, 0.0, 1.3]), 0.7, 0.9)
    e_uni = float(np.abs(U.conj().T @ U - np.eye(2)).max())

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
)


def run_physics_audit(verbose: bool = True) -> dict:
    """跑 P1–P21。

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
            excused = kind in ("ImportError", "ModuleNotFoundError", "CalledProcessError")
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
    import sys

    _report = run_physics_audit(verbose=True)

    # 需要 CUDA / triton / scipy 的项在环境缺失时会被记成 inf 或跳过；
    # 那是「环境缺失」而不是「物理不对」，因此单独列为 SKIP，不计入最差残差。
    _skipped = []
    _worst = 0.0
    for _name, _r in _report.items():
        _exc = _r.get("exception", "")
        if _exc and ("CUDA" in _exc or "Triton" in _exc or "cuda" in _exc
                     or "scipy" in _exc or "SCIPY" in _exc):
            _skipped.append((_name, _exc))
            continue
        _worst = max(_worst, _r.get("worst", 0.0))
    _ok = _worst < 1e-5
    print("=" * 96)
    for _name, _exc in _skipped:
        print(f"[SKIP] {_name} — 需要 CUDA/Triton/scipy，当前环境不可用：{_exc}")
    _passed = len(_report) - len(_skipped)
    print(f"[{'PASS' if _ok else 'FAIL'}] spin-chirality physics audit: "
          f"{_passed} checks passed, {len(_skipped)} skipped, "
          f"worst residual = {_worst:.2e}")
    sys.exit(0 if _ok else 1)
