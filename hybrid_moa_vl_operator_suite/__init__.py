"""
Unified 9-Domain Interdisciplinary Hybrid MoA Operator Suite (`hybrid_moa_vl_operator_suite`).
Pure Vision-Language Cross-Modal Fusion Edition.
Uses `if __package__:` dual-mode imports (avoiding masking symbol ImportErrors) and
resolves the real Triton kernel `persistent_moa_norm_sparse_topk_megakernel` first
with fallback to `persistent_norm_sparse_topk_agg_kernel`.
"""

if __package__:
    from . import moa_triton_kernels as _triton_mod
    from .moa_operators import (
        MoAOperatorConfig,
        NumpyMoAOperatorKernels,
        RMSNorm,
        FusedCrossModalBilinearGateFunction,
        CrossModalVisionLanguageFusionSubOp,
        HeterogeneousProposerOp,
        CrossProposalAggregatorOp,
        MixtureOfAttentionOp,
    )
    from .moa_triton_kernels import (
        KernelOptimizationReport,
        TritonTileSimulator,
        fused_cross_modal_vl_bilinear_fwd_kernel,
        fused_cross_modal_vl_bilinear_bwd_kernel,
        simulate_fused_cross_modal_vl_bilinear,
        simulate_persistent_norm_sparse_topk_megakernel,
    )
    from .quantum_tidal_operators import (
        QuantumTidalConfig,
        QuantumTidalAnalyticalKernels,
        FusedQuantumTidalWaveFunction,
        QuantumTidalFluctuationOp,
    )
    from .neuro_nonlinear_operators import (
        NeuroNonlinearConfig,
        NeuroNonlinearAnalyticalKernels,
        FusedDendriticNMDAShuntingFHNFunction,
        NonlinearDendriticNeuroOp,
    )
    from .local_self_modifying_operators import (
        LocalSelfModConfig,
        LocalSelfModifyingAnalyticalKernels,
        FusedLocalSelfModWeightFunction,
        LocalNonlinearSelfModifyingOp,
    )
    from .supermembrane_spacetime_operators import (
        SupermembraneSpacetimeConfig,
        SupermembraneAnalyticalKernels,
        FusedSupermembraneDBIWarpedFunction,
        SupermembraneSpacetimeOp,
    )
    from .tulpa_neuro_operators import (
        TulpaNeuroConfig,
        TulpaAnalyticalKernels,
        FusedTulpaCorollaryAgencyDialogueFunction,
        TulpaNeuroCognitiveOp,
    )
    from .corticospinal_chaotic_operators import (
        CorticospinalChaoticConfig,
        CorticospinalChaoticAnalyticalKernels,
        FusedCorticospinalChaoticManifoldFunction,
        CorticospinalChaoticPathwayOp,
    )
    from .baryogenesis_parity_operators import (
        ParityBaryogenesisConfig,
        ParityBaryogenesisAnalyticalKernels,
        FusedParityCPBaryogenesisFunction,
        ParityBaryogenesisAsymmetryOp,
    )
    from .dissociative_trauma_neuro_operators import (
        DissociativeTraumaConfig,
        DissociativeTraumaAnalyticalKernels,
        FusedCorticolimbicDissociativeDPDRFunction,
        DissociativeTraumaNeuroOp,
    )
    from .spinal_cord_pathway_operators import (
        SpinalCordPathwayConfig,
        SpinalCordAnalyticalKernels,
        FusedSpinalDorsalReflexRenshawFunction,
        SpinalCordPathwayOp,
    )
    from .sleep_subconscious_neuro_operators import (
        SleepSubconsciousConfig,
        SleepSubconsciousAnalyticalKernels,
        FusedSleepSubconsciousSHYSWRPGOFunction,
        SleepSubconsciousNeuroOp,
    )
    from .vacuum_zpe_fluctuation_operators import (
        VacuumZPEFluctuationConfig,
        VacuumZPEAnalyticalKernels,
        FusedVacuumZPECasimirSchwingerFunction,
        VacuumZPEFluctuationOp,
    )
    from .unified_hybrid_moa_architecture import (
        UnifiedHybridArchitectureConfig,
        UnifiedArchitectureExecutionEngine,
        UnifiedInterdisciplinaryMoALayer,
        UnifiedHybridMoABackbone,
    )
else:
    import moa_triton_kernels as _triton_mod
    from moa_operators import (
        MoAOperatorConfig,
        NumpyMoAOperatorKernels,
        RMSNorm,
        FusedCrossModalBilinearGateFunction,
        CrossModalVisionLanguageFusionSubOp,
        HeterogeneousProposerOp,
        CrossProposalAggregatorOp,
        MixtureOfAttentionOp,
    )
    from moa_triton_kernels import (
        KernelOptimizationReport,
        TritonTileSimulator,
        fused_cross_modal_vl_bilinear_fwd_kernel,
        fused_cross_modal_vl_bilinear_bwd_kernel,
        simulate_fused_cross_modal_vl_bilinear,
        simulate_persistent_norm_sparse_topk_megakernel,
    )
    from quantum_tidal_operators import (
        QuantumTidalConfig,
        QuantumTidalAnalyticalKernels,
        FusedQuantumTidalWaveFunction,
        QuantumTidalFluctuationOp,
    )
    from neuro_nonlinear_operators import (
        NeuroNonlinearConfig,
        NeuroNonlinearAnalyticalKernels,
        FusedDendriticNMDAShuntingFHNFunction,
        NonlinearDendriticNeuroOp,
    )
    from local_self_modifying_operators import (
        LocalSelfModConfig,
        LocalSelfModifyingAnalyticalKernels,
        FusedLocalSelfModWeightFunction,
        LocalNonlinearSelfModifyingOp,
    )
    from supermembrane_spacetime_operators import (
        SupermembraneSpacetimeConfig,
        SupermembraneAnalyticalKernels,
        FusedSupermembraneDBIWarpedFunction,
        SupermembraneSpacetimeOp,
    )
    from tulpa_neuro_operators import (
        TulpaNeuroConfig,
        TulpaAnalyticalKernels,
        FusedTulpaCorollaryAgencyDialogueFunction,
        TulpaNeuroCognitiveOp,
    )
    from corticospinal_chaotic_operators import (
        CorticospinalChaoticConfig,
        CorticospinalChaoticAnalyticalKernels,
        FusedCorticospinalChaoticManifoldFunction,
        CorticospinalChaoticPathwayOp,
    )
    from baryogenesis_parity_operators import (
        ParityBaryogenesisConfig,
        ParityBaryogenesisAnalyticalKernels,
        FusedParityCPBaryogenesisFunction,
        ParityBaryogenesisAsymmetryOp,
    )
    from dissociative_trauma_neuro_operators import (
        DissociativeTraumaConfig,
        DissociativeTraumaAnalyticalKernels,
        FusedCorticolimbicDissociativeDPDRFunction,
        DissociativeTraumaNeuroOp,
    )
    from spinal_cord_pathway_operators import (
        SpinalCordPathwayConfig,
        SpinalCordAnalyticalKernels,
        FusedSpinalDorsalReflexRenshawFunction,
        SpinalCordPathwayOp,
    )
    from sleep_subconscious_neuro_operators import (
        SleepSubconsciousConfig,
        SleepSubconsciousAnalyticalKernels,
        FusedSleepSubconsciousSHYSWRPGOFunction,
        SleepSubconsciousNeuroOp,
    )
    from vacuum_zpe_fluctuation_operators import (
        VacuumZPEFluctuationConfig,
        VacuumZPEAnalyticalKernels,
        FusedVacuumZPECasimirSchwingerFunction,
        VacuumZPEFluctuationOp,
    )
    from unified_hybrid_moa_architecture import (
        UnifiedHybridArchitectureConfig,
        UnifiedArchitectureExecutionEngine,
        UnifiedInterdisciplinaryMoALayer,
        UnifiedHybridMoABackbone,
    )

# Resolve real Triton kernel name (`persistent_moa_norm_sparse_topk_megakernel`) first,
# falling back to stub (`persistent_norm_sparse_topk_agg_kernel`), and expose unified public names.
persistent_moa_norm_sparse_topk_megakernel = getattr(
    _triton_mod,
    "persistent_moa_norm_sparse_topk_megakernel",
    getattr(_triton_mod, "persistent_norm_sparse_topk_agg_kernel", None),
)
persistent_norm_sparse_topk_agg_kernel = persistent_moa_norm_sparse_topk_megakernel

__all__ = [
    # 1. MoA & Vision-Language Cross-Modal Core
    "MoAOperatorConfig",
    "NumpyMoAOperatorKernels",
    "RMSNorm",
    "FusedCrossModalBilinearGateFunction",
    "CrossModalVisionLanguageFusionSubOp",
    "HeterogeneousProposerOp",
    "CrossProposalAggregatorOp",
    "MixtureOfAttentionOp",
    # 2. Triton GPU Kernels & Tile Simulator
    "KernelOptimizationReport",
    "TritonTileSimulator",
    "fused_cross_modal_vl_bilinear_fwd_kernel",
    "fused_cross_modal_vl_bilinear_bwd_kernel",
    "persistent_moa_norm_sparse_topk_megakernel",
    "persistent_norm_sparse_topk_agg_kernel",
    "simulate_fused_cross_modal_vl_bilinear",
    "simulate_persistent_norm_sparse_topk_megakernel",
    # 3. Quantum Tidal Gravitational Fluctuation
    "QuantumTidalConfig",
    "QuantumTidalAnalyticalKernels",
    "FusedQuantumTidalWaveFunction",
    "QuantumTidalFluctuationOp",
    # 4. Cortical Dendritic NMDA Shunting & FitzHugh-Nagumo
    "NeuroNonlinearConfig",
    "NeuroNonlinearAnalyticalKernels",
    "FusedDendriticNMDAShuntingFHNFunction",
    "NonlinearDendriticNeuroOp",
    # 5. Local Nonlinear Self-Modifying Fast Weights
    "LocalSelfModConfig",
    "LocalSelfModifyingAnalyticalKernels",
    "FusedLocalSelfModWeightFunction",
    "LocalNonlinearSelfModifyingOp",
    # 6. 11D Supermembrane Nambu-Poisson & DBI Spacetime
    "SupermembraneSpacetimeConfig",
    "SupermembraneAnalyticalKernels",
    "FusedSupermembraneDBIWarpedFunction",
    "SupermembraneSpacetimeOp",
    # 7. TULPA Autonomous Persona Attractor & Agency
    "TulpaNeuroConfig",
    "TulpaAnalyticalKernels",
    "FusedTulpaCorollaryAgencyDialogueFunction",
    "TulpaNeuroCognitiveOp",
    # 8. Corticospinal Chaotic Pathway & Pyragas Stabilization
    "CorticospinalChaoticConfig",
    "CorticospinalChaoticAnalyticalKernels",
    "FusedCorticospinalChaoticManifoldFunction",
    "CorticospinalChaoticPathwayOp",
    # 9. Parity Violation & Baryogenesis Asymmetry
    "ParityBaryogenesisConfig",
    "ParityBaryogenesisAnalyticalKernels",
    "FusedParityCPBaryogenesisFunction",
    "ParityBaryogenesisAsymmetryOp",
    # 10. Dissociative Trauma & Corticolimbic Defense
    "DissociativeTraumaConfig",
    "DissociativeTraumaAnalyticalKernels",
    "FusedCorticolimbicDissociativeDPDRFunction",
    "DissociativeTraumaNeuroOp",
    # 11. Spinal Cord Dorsal Horn & Renshaw Pathway
    "SpinalCordPathwayConfig",
    "SpinalCordAnalyticalKernels",
    "FusedSpinalDorsalReflexRenshawFunction",
    "SpinalCordPathwayOp",
    # 12. Human Sleep Physiology & Deep Subconscious Neuroscience
    "SleepSubconsciousConfig",
    "SleepSubconsciousAnalyticalKernels",
    "FusedSleepSubconsciousSHYSWRPGOFunction",
    "SleepSubconsciousNeuroOp",
    # 13. Quantum Fluctuation & Vacuum Zero-Point Energy (ZPE)
    "VacuumZPEFluctuationConfig",
    "VacuumZPEAnalyticalKernels",
    "FusedVacuumZPECasimirSchwingerFunction",
    "VacuumZPEFluctuationOp",
    # 14. Unified Assembly Architecture
    "UnifiedHybridArchitectureConfig",
    "UnifiedArchitectureExecutionEngine",
    "UnifiedInterdisciplinaryMoALayer",
    "UnifiedHybridMoABackbone",
]
