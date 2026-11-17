# -*- coding: utf-8 -*-
"""`python -m spin_chirality` → 运行 P1–P45 物理自检。

缺 torch / triton / CUDA 工具链 / scipy 的项自动标"跳过"、不算失败：
只装 numpy 也能跑完并退出 0。
"""

from __future__ import annotations

import sys

from .operator import run_physics_audit

if __name__ == "__main__":
    report = run_physics_audit(verbose=True)
    live = [r["worst"] for r in report.values() if not r.get("skipped")]
    worst = max(live) if live else 0.0
    n_skip = sum(1 for r in report.values() if r.get("skipped"))
    print(f"共 {len(report)} 项（跳过 {n_skip} 项），最大残差 {worst:.2e}")
    sys.exit(0 if worst < 1e-5 else 1)
