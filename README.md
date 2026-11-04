# Neuro-Physical Hierarchical MoA Backbone

A 12-module neural operator library for **Mixture-of-Architectures (MoA)** backbones.
Each module translates one domain — particle physics, systems neuroscience, quantum
fluctuation, spacetime geometry — into end-to-end differentiable PyTorch operators,
fused Triton GPU kernels, and analytical NumPy reference kernels.

Total: **10,619 lines** of operator code (12 library modules) + a 386-line
clean-room smoke test.

---

## Modules

| Module | Domain | Lines |
|---|---|---|
| `unified_hybrid_moa_architecture.py` | 9-domain master backbone, 6-stage multi-verification + 5-fold cross-validation | 1,157 |
| `moa_operators.py` | System-level MoA: heterogeneous proposers (Attention / SSM / Conv / SwiGLU), zero-OOM selective SSM, sparse router, cross-proposal attention | 1,142 |
| `corticospinal_chaotic_operators.py` | Brain–spinal bidirectional pathway, multivariable chaotic regulation, Pyragas closed-loop stabilisation | 922 |
| `spinal_cord_pathway_operators.py` | Spinal segmental pathways, dorsal reflex + Renshaw recurrence | 920 |
| `baryogenesis_parity_operators.py` | Parity non-conservation, CP-violating phase interference, Sakharov conditions, matter–antimatter asymmetry | 882 |
| `supermembrane_spacetime_operators.py` | High-dimensional warped projection spacetime, DBI membrane dynamics | 848 |
| `quantum_tidal_operators.py` | Quantum tidal fluctuation, WKB tunnelling, decoherence | 846 |
| `dissociative_trauma_neuro_operators.py` | Lanius bimodal corticolimbic regulation, DPDR thalamocortical gating, PTSD | 782 |
| `local_self_modifying_operators.py` | Local nonlinear weight self-modification: bounded rational plasticity, Oja homeostasis, delta-Oja fast weights | 780 |
| `neuro_nonlinear_operators.py` | Dendritic NMDA + shunting inhibition, FitzHugh–Nagumo soma, short-term plasticity | 777 |
| `tulpa_neuro_operators.py` | Tulpa / autonomous sub-agent cognition: corollary discharge, TPJ agency attribution, DMN dialogue | 676 |
| `moa_triton_kernels.py` | Triton kernel suite + deterministic tile-by-tile SRAM simulator | 501 |

## API conventions

Every module follows the same four-layer layout, so the layers can be mixed and
matched:

1. `*Config` — dataclass of all hyper-parameters (defaults are runnable as-is).
2. `*AnalyticalKernels` / `NumpyMoAOperatorKernels` — pure NumPy static methods:
   forward, backward, and scan variants. These are the reference implementation and
   need **no GPU and no torch**.
3. `*Op` — `torch.nn.Module` wrappers (`HeterogeneousProposerOp`,
   `MixtureOfAttentionOp`, `CrossProposalAggregatorOp`, `NonlinearDendriticNeuroOp`, …).
   `forward()` returns `(out, diagnostics)` or `[B, N, L, D]` per-branch outputs.
4. `Fused*Function` — `torch.autograd.Function` implementations of the fused
   forward/backward, plus `@triton.jit` kernels where Triton is available.

Each module also ships a self-contained verification entry point
(`run_*_verification*`), and `unified_hybrid_moa_architecture.py` runs a 6-stage
multi-verification with 5-fold cross-validation across all nine domains.

## Dependencies

| Layer | Requirement |
|---|---|
| NumPy analytical kernels, tile simulator | `numpy` only |
| torch operators | `torch` (CPU is enough) |
| Triton kernels | `triton` + CUDA GPU (optional — the simulator covers CPU verification) |

Nothing is imported at module load time beyond `numpy`, `math`, `dataclasses` and
`typing`; torch/triton are guarded by `HAS_TORCH` / `HAS_TRITON` flags.

## Quick start

```python
from moa_operators import MoAOperatorConfig, HeterogeneousProposerOp

cfg = MoAOperatorConfig(d_model=256, num_proposers=4, top_k_proposers=2,
                        proposer_mode="heterogeneous")
op = HeterogeneousProposerOp(cfg)          # [B, L, D] -> [B, N, L, D]
```

NumPy-only reference path (no torch needed):

```python
from moa_operators import NumpyMoAOperatorKernels

k = NumpyMoAOperatorKernels()
y, inv_rms = k.branch_rmsnorm_forward(x, gamma)          # x: [B, L, D], gamma: [D]
y_seq, _ = k.ssm_sequential_scan_ref(dA, dB_u, C)        # dA, dB_u: [B, L, D, N]
mask, stats = k.build_mixture_of_attention_mask(seq_len=128, n_heads=8)
```

## Verification

`smoke_test.py` is a clean-room test: fresh interpreter, CPU only, no Triton.

```bash
python smoke_test.py            # everything available
python smoke_test.py --quick    # imports + numpy kernels only
```

Verified on `numpy 2.2.6` + torch (CPU), 9/9 checks:

- 12/12 modules import under their canonical snake-case names
- branch RMSNorm forward/backward, finite-difference gradient check (rel-err 2.7e-09)
- 4 SSM scan implementations (sequential / Kogge-Stone / chunkwise SSD / zero-OOM
  checkpointed) agree with the sequential reference to 5.6e-17
- per-head MoA attention mask: sink+window / dilated / full-causal heads, all causal
- sparse top-k router + cross-proposal attention (attention sums to 1 over proposers)
- fused per-proposer GEMM forward/backward with gradient check
- torch operators run forward + backward (`HeterogeneousProposerOp`,
  `MixtureOfAttentionOp`, both aggregator modes, two domain operators)
- Triton tile simulators reproduce their analytical references to 1.8e-15

## Notes

- This repository holds the **original version** of the operator library. The
  adapted copies used inside the
  [hetero-moe](https://github.com/Microbiosis/hetero-moe) research line
  (`research/aligner/v62_dimension_projection_space/moa_operators.py` and friends)
  are modified for that project and are maintained there.

## License

[Apache License 2.0](LICENSE) — Copyright 2026 Microbiosis.

- **Provenance.** The 12 operator modules were written to specifications provided
  by the copyright holder, with implementation assistance from Google Gemini.
- **No per-file headers.** License headers were deliberately not inserted into the
  operator modules so they stay byte-identical to the original versions; the
  repository-level `LICENSE` applies to every file here.
- **Patents.** Apache-2.0 includes an express patent grant (§3) and a defensive
  termination clause, which is why it was chosen over MIT.
- **Contributions.** Per §5, anything intentionally submitted for inclusion is
  under this same license unless stated otherwise.
- **Trademarks.** §6 grants no right to use the Microbiosis name or marks.

