# -*- coding: utf-8 -*-
"""二次量子化的最小实现：Fock 空间、反对易、Pauli 排斥、多体 vs Slater 行列式。

只处理**有限个模式**（m ≤ 10，Fock 维数 2^m ≤ 1024），矩阵是稠密的——够用来判定
“多体演化 = 单粒子演化的 Slater 行列式”这一类**精确**关系，不适合做大 N 的模拟。

算符约定（占据数基，比特 i = 模式 i 是否占据）：
    c†_i |n⟩ = (−1)^{#(i 以下已占据)} |n + e_i⟩   （若 i 已占据则为 0）
    c_i  |n⟩ = (−1)^{#(i 以下已占据)} |n − e_i⟩   （若 i 空则为 0）
这个符号约定保证 {c_i, c†_j} = δ_ij、{c_i, c_j} = 0。
"""
from __future__ import annotations

import numpy as np


def _parity_sign(n: int, i: int) -> int:
    """占据数 n 中，第 i 位以下有多少个占据 → 交换符号。"""
    return -1 if bin(n & ((1 << i) - 1)).count("1") % 2 else 1


def annihilation(n_modes: int) -> list:
    """返回 [c_0, …, c_{m−1}]，每个是 2^m × 2^m 的稠密矩阵。"""
    dim = 1 << n_modes
    ops = []
    for i in range(n_modes):
        M = np.zeros((dim, dim), dtype=complex)
        for n in range(dim):
            if (n >> i) & 1:                       # 该模式已占据 → 可以湮灭
                m = n ^ (1 << i)
                M[m, n] = _parity_sign(n, i)
        ops.append(M)
    return ops


def creation(n_modes: int) -> list:
    """返回 [c†_0, …, c†_{m−1}]（= 湮灭算符的转置共轭）。"""
    return [a.conj().T for a in annihilation(n_modes)]


def number_operator(n_modes: int, i: int) -> np.ndarray:
    a = annihilation(n_modes)[i]
    return a.conj().T @ a


def many_body_hamiltonian(H1: np.ndarray) -> np.ndarray:
    """把单粒子矩阵 H1（m×m）提升成多体哈密顿量 Σ_ij H1_ij c†_i c_j。"""
    m = H1.shape[0]
    c = annihilation(m)
    cd = creation(m)
    H = np.zeros((1 << m, 1 << m), dtype=complex)
    for i in range(m):
        for j in range(m):
            if abs(H1[i, j]) > 0.0:
                H += H1[i, j] * (cd[i] @ c[j])
    return H


def occupancy_state(n_modes: int, occupied) -> np.ndarray:
    """给定占据模式列表，返回对应的 Fock 基矢。"""
    n = 0
    for i in occupied:
        n |= 1 << i
    v = np.zeros(1 << n_modes, dtype=complex)
    v[n] = 1.0
    return v


def check_canonical(m: int = 5) -> dict:
    """P27(a)：{c_i, c†_j} = δ_ij、{c_i, c_j} = 0、({c†,c†} = 0)。"""
    c = annihilation(m)
    cd = creation(m)
    e_anti = 0.0
    e_cd = 0.0
    e_cc = 0.0
    for i in range(m):
        for j in range(m):
            e_anti = max(e_anti, float(np.abs(c[i] @ cd[j] + cd[j] @ c[i]
                                               - (np.eye(1 << m) if i == j else 0)).max()))
            e_cd = max(e_cd, float(np.abs(cd[i] @ cd[j] + cd[j] @ cd[i]).max()))
            e_cc = max(e_cc, float(np.abs(c[i] @ c[j] + c[j] @ c[i]).max()))
    return {"{c_i,c†_j} = δ_ij": e_anti, "{c†,c†} = 0": e_cd, "{c,c} = 0": e_cc}


def check_pauli(m: int = 5) -> dict:
    """P27(b)：Pauli 排斥——同一模式不能占两次；(c†)² = 0，且 n_i 只有 0/1 本征值。"""
    c = annihilation(m)
    cd = creation(m)
    e_double = max(float(np.abs(cd[i] @ cd[i]).max()) for i in range(m))
    n_op = cd[2] @ c[2]
    ev = np.linalg.eigvalsh(n_op)
    e_spec = float(max(np.abs(ev - np.round(np.clip(ev, 0, 1))).max(), 0.0))
    # 填充数上限：把模式 2 占两次也还是 1
    v = occupancy_state(m, [2])
    e_fill = float(np.abs(np.linalg.norm(cd[2] @ cd[2] @ v)))
    return {"(c†)² = 0": e_double, "n_i 本征值 ∈ {0,1}": e_spec, "重复占据被禁止": e_fill}


def check_slater(m: int = 6, seed: int = 5) -> dict:
    """P27(c)：多体演化 = 单粒子演化的 Slater 行列式（二次量子化的核心等式）。

    取随机哈密顿量 H1，先在单粒子层面演化出 U = e^{−iH1 t}，再在多体层面演化初态
    （占据 modes 0,1），比较每个模式的 ⟨n_i⟩：
        多体：⟨n_i⟩ = ⟨ψ_MB| c†_i c_i |ψ_MB⟩
        单粒子：⟨n_i⟩ = Σ_{a ∈ 占据} |U_{ia}|²
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    H1 = (A + A.conj().T) / 2.0
    t = 0.37
    w, v = np.linalg.eigh(H1)
    U = (v * np.exp(-1j * w * t)) @ v.conj().T

    Hmb = many_body_hamiltonian(H1)
    wm, vm = np.linalg.eigh(Hmb)
    occ = [0, 1]
    psi0 = occupancy_state(m, occ)
    psi_t = (vm * np.exp(-1j * wm * t)) @ (vm.conj().T @ psi0)

    err = 0.0
    for i in range(m):
        n_op = number_operator(m, i)
        mb = float(np.real(psi_t.conj() @ n_op @ psi_t))
        sp = float(sum(abs(U[i, a]) ** 2 for a in occ))
        err = max(err, abs(mb - sp))

    # 粒子数守恒
    Ntot = sum(number_operator(m, i) for i in range(m))
    e_num = abs(float(np.real(psi_t.conj() @ Ntot @ psi_t)) - len(occ))
    return {"⟨n_i⟩ 多体 vs Slater": err, "总粒子数守恒": e_num}


if __name__ == "__main__":
    import json

    print(json.dumps(check_canonical(), ensure_ascii=False, indent=2))
    print(json.dumps(check_pauli(), ensure_ascii=False, indent=2))
    print(json.dumps(check_slater(), ensure_ascii=False, indent=2))
