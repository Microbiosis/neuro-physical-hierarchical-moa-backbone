# -*- coding: utf-8 -*-
"""spin_chirality —— 物理忠实的旋量手性算子。

一个模块、一份实现（`operator.py`）：

    可微算子（torch + 可选 Triton 融合核）用于训练与序列传播；同一批物理对象的
    NumPy 写法（一步前向、传播子、序列扫描）就在同一文件里，供逐项对照；单粒子
    QFT 工具（Dirac 哈密顿量、Larmor 酉演化、1+1 维磁通指标）也在其中。原先另有
    一份 qft.py 与 operator 平行维护，2026-11-10 已合并，不再有第二套实现。

公开对象：手性投影 P_{L/R}、γ 矩阵与 γ⁵、SU(2) 手性规范（左右独立）、
Larmor 进动、Dirac 质量项、非厄米吸收、轴矢流 j⁵ 与手性荷 Q5、
以及 ABJ 反常的积分记账接口。

快速开始::

    import torch
    from spin_chirality import SpinChiralityConfig, SpinChiralityOperator

    op = SpinChiralityOperator(SpinChiralityConfig(dim=64, dt=0.05))
    x = torch.randn(2, 16, 64, dtype=torch.complex128)    # ℂ^{B×L×4G}
    y = op(x)
    y.real.sum().backward()

物理自检（只装 numpy 即可跑完；torch / triton / scipy 都是可选）::

    from spin_chirality import run_physics_audit
    report = run_physics_audit()          # P1–P45（缺可选依赖的项自动跳过）

3+1D 格点指标::

    from spin_chirality import lattice4d            # 需要 scipy
    lattice4d.check_index_4d(N=8, R=2.5)            # FLSG 单瞬子 → index = −1
"""

from __future__ import annotations

from .operator import (
    RealSpinChiralityOperator,
    SpinChiralityConfig,
    SpinChiralityOperator,
    fit_chiral_procrustes,
    fit_su2_procrustes,
    real_spin_chirality_numpy_forward,
    anomaly_axial_charge_step,
    apply_su2_spinor,
    check_flux_index,
    check_spectral_flow,
    dirac_hamiltonian,
    ff_from_eb,
    gauge_link,
    gauge_links,
    larmor_unitary,
    propagator_numpy,
    propagator_sequence_numpy,
    quat_apply,
    quat_conj,
    quat_mul,
    run_physics_audit,
    sequence_numpy_forward,
    sequence_scan_numpy,
    spin_chirality_numpy_forward,
    spin_expectation_numpy,
    su2,
    su2_exp,
    su2_generator,
    su2_quat,
)
from . import fock
from . import kogut_susskind
from . import ks_basis
from . import ks_lanczos
from . import ks_mps
from .ks_photon import corr_electric_connected, gap_flux
from . import ks_photon
from .ks_photon_weight import field_matrix_elements, photon_weight
from . import ks_photon_weight
from .ks_photon_mass import dense_photon_weights, fit_m_gamma, photon_mass_point
from . import ks_photon_mass
from .ks_dispersion import closure_gate, dispersion_point, fit_g0_forms
from . import ks_dispersion
from .ks_omega import dispersion, extrapolate, z_local
from . import ks_omega
from . import lattice1d
from . import lattice4d
from . import schwinger

__version__ = "0.2.0"

__all__ = [
    "SpinChiralityConfig",
    "SpinChiralityOperator",
    "RealSpinChiralityOperator",
    "spin_chirality_numpy_forward",
    "real_spin_chirality_numpy_forward",
    "fit_chiral_procrustes",
    "fit_su2_procrustes",
    "su2",
    "su2_exp",
    "su2_generator",
    "su2_quat",
    "apply_su2_spinor",
    "quat_apply",
    "quat_conj",
    "quat_mul",
    "propagator_numpy",
    "propagator_sequence_numpy",
    "sequence_numpy_forward",
    "sequence_scan_numpy",
    "run_physics_audit",
    "ff_from_eb",
    "gauge_link",
    "gauge_links",
    "anomaly_axial_charge_step",
    "dirac_hamiltonian",
    "larmor_unitary",
    "spin_expectation_numpy",
    "check_flux_index",
    "check_spectral_flow",
    "fock",
    "kogut_susskind",
    "ks_basis",
    "ks_lanczos",
    "ks_mps",
    "ks_photon",
    "gap_flux",
    "corr_electric_connected",
    "ks_photon_weight",
    "photon_weight",
    "field_matrix_elements",
    "ks_photon_mass",
    "photon_mass_point",
    "dense_photon_weights",
    "fit_m_gamma",
    "ks_dispersion",
    "dispersion_point",
    "closure_gate",
    "fit_g0_forms",
    "ks_omega",
    "dispersion",
    "z_local",
    "extrapolate",
    "lattice1d",
    "lattice4d",
    "schwinger",
    "__version__",
]
