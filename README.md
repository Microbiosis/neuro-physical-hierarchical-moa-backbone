# Neuro-Physical Hierarchical MoA Backbone

**Hybrid MoA Vision-Language Operator Suite** — 12-module neural operator library for
Mixture-of-Architectures (MoA) backbones, in the **pure Vision-Language cross-modal
fusion edition** (the legacy 1-D causal SSM branch has been removed).

Every module ships four layers: a dataclass config, deterministic NumPy analytical
kernels, `torch.nn.Module` operators, and fused `autograd.Function` / `@triton.jit`
implementations — and every layer is importable even without torch installed
(fallback stubs raise a clear `ImportError` only when instantiated).

Total: **9,742 lines** of operator code across 12 modules
(+ 249 lines of package `__init__.py` = 9,991 lines of Python), plus a 482-line
clean-room smoke test.

---

## Layout

```
.
├── LICENSE / NOTICE                  Apache-2.0 + provenance + fix log
├── smoke_test.py                     clean-room verification (13/13)
├── README.md                         本文件
└── hybrid_moa_vl_operator_suite/     算子包（Python package）
    ├── __init__.py                   55 个公开导出（Config / Kernels / Op / Function / Triton 内核）
    ├── README.md                     包内说明（原始版本文档）
    ├── moa_operators.py              系统级 MoA + 跨模态图文融合（978 行）
    ├── moa_triton_kernels.py         Triton 内核 + tile 模拟器（386 行）
    ├── unified_hybrid_moa_architecture.py   九域总装 + 验证引擎（1,060 行）
    └── … 9 个领域算子模块
```

Two import styles are supported:

```python
# 1) 作为包导入（推荐）
from hybrid_moa_vl_operator_suite import MoAOperatorConfig, NumpyMoAOperatorKernels
from hybrid_moa_vl_operator_suite.moa_operators import CrossModalVisionLanguageFusionSubOp

# 2) 直接在包目录里跑单个模块的验证套件
#    python3 hybrid_moa_vl_operator_suite/unified_hybrid_moa_architecture.py
```

## Modules

| Module | Domain | Lines |
|---|---|---|
| `unified_hybrid_moa_architecture.py` | 九域总装层 `UnifiedInterdisciplinaryMoALayer` + 多层 `UnifiedHybridMoABackbone` + 执行引擎 + LODO 消融 | 1,060 |
| `moa_operators.py` | 系统级 MoA：异构 proposer / 逐头注意力 / 聚合 + **跨模态图文融合** | 978 |
| `corticospinal_chaotic_operators.py` | 皮质-脊髓混沌通路、Pyragas 时滞反馈镇定 | 938 |
| `spinal_cord_pathway_operators.py` | 脊髓节段通路、Melzack-Wall 门控、Renshaw 递归 | 936 |
| `baryogenesis_parity_operators.py` | 宇称不守恒、CKM 相位干涉、正反物质湮灭余量 | 898 |
| `supermembrane_spacetime_operators.py` | 11 维超膜 DBI、Randall-Sundrum 翘曲 KK 模态 | 864 |
| `quantum_tidal_operators.py` | WKB 隧穿、Schwarzschild 四极潮汐张量 | 862 |
| `neuro_nonlinear_operators.py` | 树突 NMDA 镁门控、FitzHugh-Nagumo、Kuramoto 相位耦合 | 793 |
| `local_self_modifying_operators.py` | Delta-Oja 在线可塑性、奇异值能量软裁剪 | 796 |
| `tulpa_neuro_operators.py` | 多子人格吸引子、精度加权自由能路由、Agency 比较器 | 692 |
| `dissociative_trauma_neuro_operators.py` | Stage-I 前门控：杏仁核超唤醒、PAG 过载关断、ANP/EP 隔离 + 独立验证套件 | 539 |
| `moa_triton_kernels.py` | Triton 融合内核 + 逐 tile SRAM 模拟器 | 386 |

## What this edition contains

- **Expert 0 = 跨模态图文融合** `CrossModalVisionLanguageFusionSubOp`
  （双向 2D 空间邻域卷积 + `R_anchor` 共享潜锚点共注意力 + Hadamard-Bilinear 门控合成），
  替代原来的 1-D 因果 SSM 分支；旧 SSM 及其别名映射已整体移除。
- **零距离衰减**：任意图像 patch 与任意文本 token 通过 `R_anchor` 个共享锚点直接交互，
  复杂度 `O(L·R_anchor·D)`，没有指数衰减项。
- **多层骨干** `UnifiedHybridMoABackbone`：堆叠 N 个 `UnifiedInterdisciplinaryMoALayer`
  （残差 + final RMSNorm），返回逐层遥测。
- **无 torch 也可 import**：每个模块在 `not HAS_TORCH` 时定义 stub 类
  （`RMSNorm` / `Fused*Function` / `*Op`），实例化时才抛 `ImportError` 并提示改用
  `AnalyticalKernels`；numpy 路径完全不依赖 torch。
- **包结构**：12 个模块收进 `hybrid_moa_vl_operator_suite/`，`__init__.py` 导出 55 个符号，
  支持包导入与平目录导入两种方式。

### 融合算子的数学形式

```
# 1) 双向低秩跨模态锚点共注意力（零距离衰减）
z_anchor[r]   = Σ_l softmax_l(anchor[r]·(v+t)/√D) · v[l]        # 汇成 R 个锚点
c_align[l]    = Σ_r softmax_r(t[l]·z_anchor[r]) · z_anchor[r]   # 广播回每个 token

# 2) Hadamard-Bilinear 跨模态门控合成
G_cm     = sigmoid(α_vis·V + α_txt·T)
Y_interp = G_cm·V + (1-G_cm)·T
Out      = Y_interp + β_bilin·(V⊙T + C_align) / sqrt(1 + λ·(V²+T²))
```

## Dependencies

| Layer | Requirement |
|---|---|
| NumPy analytical kernels, tile simulator | `numpy` only |
| torch operators | `torch` (CPU is enough) |
| Triton kernels | `triton` + CUDA GPU (optional — the simulator covers CPU verification) |

## Quick start

```python
import torch
from hybrid_moa_vl_operator_suite.moa_operators import (
    CrossModalVisionLanguageFusionSubOp, MoAOperatorConfig, HeterogeneousProposerOp)

# 跨模态图文融合：x [B, L, D] -> fused [B, L, D]
fusion = CrossModalVisionLanguageFusionSubOp(d_model=256, num_anchors=16)
y = fusion(torch.randn(2, 64, 256))

# 系统级 MoA：异构 proposer -> [B, N, L, D] -> top-k 聚合
cfg = MoAOperatorConfig(d_model=256, num_proposers=8, top_k_proposers=3)
proposals = HeterogeneousProposerOp(cfg)(torch.randn(2, 64, 256))

# 多层骨干
from hybrid_moa_vl_operator_suite import UnifiedHybridArchitectureConfig, UnifiedHybridMoABackbone
bcfg = UnifiedHybridArchitectureConfig(d_model=256, n_layers=4)
out, aux, telemetries = UnifiedHybridMoABackbone(bcfg)(torch.randn(2, 64, 256))
```

NumPy-only reference path:

```python
from hybrid_moa_vl_operator_suite import NumpyMoAOperatorKernels

k = NumpyMoAOperatorKernels()
c_align, cache = k.cross_modal_anchor_co_attention_forward(v_vis, t_txt, anchors)  # 零距离衰减
out, cache = k.cross_modal_vl_bilinear_fusion_forward(
    v_vis, t_txt, c_align, alpha_vis, alpha_txt, beta_bilin, lam_norm=0.35)
```

## Verification

`smoke_test.py` is the clean-room test: fresh interpreter, CPU only, no Triton.

```bash
python smoke_test.py            # everything available
python smoke_test.py --quick    # imports + numpy kernels only
```

Result: **13/13 checks passed** (numpy 2.2.6 + torch 2.10.0 CPU).

| Check | Result |
|---|---|
| package imports (12 modules, 55 exports) | 12/12 |
| branch RMSNorm fwd/bwd + finite-difference gradient | fwd 1.8e-15, grad 2.7e-09 |
| cross-modal anchor co-attention (zero decay, both attentions normalised) | 0.0e+00, head/tail = 1.003 |
| VL Hadamard-bilinear fusion: closed form, β=0 degeneration, gate saturation, backward gradient | fwd 8.9e-16, grad 7.7e-11 |
| per-head MoA attention mask (causal, 3 densities 0.198/0.296/0.516) | ok |
| sparse top-k router | aux 0.0222 |
| torch `CrossModalVisionLanguageFusionSubOp` fwd/bwd | 13.4K params, grad reaches input |
| torch MoA stack (proposer + attention + 2 aggregator modes) | density 0.913 |
| torch `UnifiedInterdisciplinaryMoALayer` (8 experts) | 372.6K params, grad 0.118 |
| torch `UnifiedHybridMoABackbone` (3 layers) | 1,118.0K params, grad reaches input |
| torch domain operators (quantum tidal, neuro) | both return (out, diagnostics) |
| upstream dissociative verification suite | 7 gradchecks PASS, shock output bounded |
| Triton tile simulators vs analytical references | 2.2e-16 / 8.9e-16 |

Upstream benchmarks shipped with the package (`python3 <module>.py` inside the package):

- `FusedCrossModalVLBilinear`: diff **0.00e+00**, launches 12 → 1, **HBM traffic −80.5%**
- `PersistentNormSparseTopKMegakernel` (top-2/4): diff 2.66e-15, launches 17 → 1, **HBM −90.7%**
- Image-text alignment R²: unaligned additive baseline **+0.7047** → VL fusion branch **+0.9990**
- LODO cross-ablation: all 8 experts active, relative output impact 6.52%–7.12%;
  unified representation effective rank **102.52 / 128**
- Dissociative trauma gradcheck: 7 variables, relative L2 error 6.7e-11 – 7.5e-10;
  extreme +50.0 shock output bounded at 1.32 (< 5.0), PAG shutdown gate 7.1e-218

## Notes

- This repository holds the **original version** of the operator library. Adapted copies
  used inside [hetero-moe](https://github.com/Microbiosis/hetero-moe)
  (`research/aligner/v62_dimension_projection_space/`) are modified for that project
  and are maintained there.
- Two files were patched on top of the shipped archive so the package imports at all —
  see `NOTICE` for exactly what changed: `__init__.py` imported a Triton kernel name
  that only exists when Triton is **absent**, and
  `unified_hybrid_moa_architecture.py` used bare imports plus a `sys.path` hack that
  loaded a second copy of every module.

## License

[Apache License 2.0](LICENSE) — Copyright 2026 Microbiosis.

Provenance: written to specifications provided by the copyright holder, with
implementation assistance from Google Gemini. Per-module license headers were
deliberately not inserted so the operator files stay byte-identical to the shipped
archive (except the two import fixes noted above); the repository-level `LICENSE`
applies to every file here. Apache-2.0 §3 grants patent rights with defensive
termination; §6 grants no trademark rights.
