"""
Unified 9-Domain Interdisciplinary Hybrid MoA Operator Suite (`hybrid_moa_vl_operator_suite`).
Pure Vision-Language Cross-Modal Fusion Edition (Zero Legacy 1D SSM Dependencies).
"""

from .moa_operators import (
    MoAOperatorConfig,
    NumpyMoAOperatorKernels,
)
from .moa_triton_kernels import (
    KernelOptimizationReport,
    TritonTileSimulator,
)
from .quantum_tidal_operators import (
    QuantumTidalConfig,
    QuantumTidalAnalyticalKernels,
)
from .neuro_nonlinear_operators import (
    NeuroNonlinearConfig,
    NeuroNonlinearAnalyticalKernels,
)
from .local_self_modifying_operators import (
    LocalSelfModConfig,
    LocalSelfModifyingAnalyticalKernels,
)
from .supermembrane_spacetime_operators import (
    SupermembraneSpacetimeConfig,
    SupermembraneAnalyticalKernels,
)
from .spinal_cord_pathway_operators import (
    SpinalCordPathwayConfig,
    SpinalCordAnalyticalKernels,
)
from .tulpa_neuro_operators import (
    TulpaNeuroConfig,
    TulpaAnalyticalKernels,
)
from .dissociative_trauma_neuro_operators import (
    DissociativeTraumaConfig,
    DissociativeTraumaAnalyticalKernels,
)
from .corticospinal_chaotic_operators import (
    CorticospinalChaoticConfig,
    CorticospinalChaoticAnalyticalKernels,
)
from .baryogenesis_parity_operators import (
    ParityBaryogenesisConfig,
    ParityBaryogenesisAnalyticalKernels,
)
from .unified_hybrid_moa_architecture import (
    UnifiedHybridArchitectureConfig,
    UnifiedArchitectureExecutionEngine,
)

__all__ = [
    # 系统级 MoA / 跨模态图文融合
    "MoAOperatorConfig",
    "NumpyMoAOperatorKernels",
    # Triton 内核模拟
    "KernelOptimizationReport",
    "TritonTileSimulator",
    # 九域 Config + 解析内核
    "QuantumTidalConfig",
    "QuantumTidalAnalyticalKernels",
    "NeuroNonlinearConfig",
    "NeuroNonlinearAnalyticalKernels",
    "LocalSelfModConfig",
    "LocalSelfModifyingAnalyticalKernels",
    "SupermembraneSpacetimeConfig",
    "SupermembraneAnalyticalKernels",
    "SpinalCordPathwayConfig",
    "SpinalCordAnalyticalKernels",
    "TulpaNeuroConfig",
    "TulpaAnalyticalKernels",
    "DissociativeTraumaConfig",
    "DissociativeTraumaAnalyticalKernels",
    "CorticospinalChaoticConfig",
    "CorticospinalChaoticAnalyticalKernels",
    "ParityBaryogenesisConfig",
    "ParityBaryogenesisAnalyticalKernels",
    # 统一架构
    "UnifiedHybridArchitectureConfig",
    "UnifiedArchitectureExecutionEngine",
]
