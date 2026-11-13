"""
Interdisciplinary Hybrid MoA Operator Suite (`hybrid_moa_vl_operator_suite`).

25 standalone operator modules; there is no orchestration layer - every module is
independently importable and carries its own verification suite.

Zero-hardcoded-table dynamic package initializer:
1. Automatically discovers all `.py` submodules in the package directory via `pkgutil.iter_modules`
   (ordering standalone operator/kernel modules before the verification module).
2. Dynamically scans each submodule's `__all__` (or inspects objects whose `__module__ == mod.__name__`)
   and resolves every symbol via `getattr` with fault-tolerant fallback to nested `*Kernels`,
   `*Simulator`, and `*Engine` classes.
3. Provides a PEP 562 module-level `__getattr__` hook for on-demand cross-submodule lookup.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
from types import ModuleType
from typing import Dict, List

_MISSING = object()
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# 上游的 payload 族用顶层静态 import（`import payload_core as core`）互相引用，
# 这在「包目录在 sys.path 上」时成立。作为 `hybrid_moa_vl_operator_suite.xxx`
# 被导入时顶层名不在 sys.path，会 ModuleNotFoundError —— 而自动发现的容错分支会
# 把这个错误吞掉，导致这些模块静默没有加载。
# 因此把包目录放进 sys.path：顶层 import 能解析，模块也能正常注册到包命名空间下。
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
_LOADED_MODULES: Dict[str, ModuleType] = {}
__all__: List[str] = []


def _discover_submodule_names() -> List[str]:
    """Automatically discover all submodule names in `_PKG_DIR` without a hardcoded file list."""
    names = [
        info.name
        for info in pkgutil.iter_modules([_PKG_DIR])
        if not info.ispkg and not info.name.startswith("_")
    ]

    def _sort_key(name: str) -> tuple:
        if name.startswith("moa_"):
            return (0, name)
        if name.endswith(("_operators", "_kernels")):
            return (1, name)
        if "architecture" in name:
            return (2, name)
        return (3, name)

    return sorted(names, key=_sort_key)


def _load_submodule(mod_name: str) -> ModuleType:
    if __package__:
        return importlib.import_module(f".{mod_name}", __package__)
    return importlib.import_module(mod_name)


def _discover_module_symbols(mod: ModuleType) -> List[str]:
    """Read `mod.__all__` via `getattr`, or fall back to objects defined in `mod.__name__`."""
    explicit_all = getattr(mod, "__all__", None)
    if isinstance(explicit_all, (list, tuple)):
        return [str(s) for s in explicit_all if isinstance(s, str) and not s.startswith("_")]

    discovered: List[str] = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_") or isinstance(obj, ModuleType):
            continue
        if getattr(obj, "__module__", None) == mod.__name__ or "triton" in str(type(obj)).lower():
            discovered.append(name)
    return discovered


def _resolve_symbol_from_module(mod: ModuleType, name: str) -> object:
    """Fault-tolerantly resolve `name` from `mod` or its nested `*Kernels`/`*Simulator`/`*Engine` classes."""
    val = getattr(mod, name, _MISSING)
    if val is not _MISSING:
        return val
    for _, cls_obj in inspect.getmembers(mod, inspect.isclass):
        if getattr(cls_obj, "__module__", None) == mod.__name__:
            nested = getattr(cls_obj, name, _MISSING)
            if nested is not _MISSING:
                return nested
    return _MISSING


def _bootstrap_package_namespace() -> None:
    g = globals()
    for mod_name in _discover_submodule_names():
        try:
            mod = _load_submodule(mod_name)
        except Exception:
            continue
        _LOADED_MODULES[mod_name] = mod
        g[mod_name] = mod

        for sym in _discover_module_symbols(mod):
            val = _resolve_symbol_from_module(mod, sym)
            if val is _MISSING:
                continue
            if sym in g and sym in __all__:
                continue
            g[sym] = val
            if sym not in __all__:
                __all__.append(sym)

        for _, cls_obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(cls_obj, "__module__", None) == mod.__name__:
                for method_name in dir(cls_obj):
                    if method_name.startswith("simulate_") and method_name not in g:
                        m_val = getattr(cls_obj, method_name, _MISSING)
                        if m_val is not _MISSING:
                            g[method_name] = m_val
                            if method_name not in __all__:
                                __all__.append(method_name)


_bootstrap_package_namespace()


def __getattr__(name: str) -> object:
    """PEP 562 dynamic fallback lookup across all loaded submodules and their classes."""
    if name in globals():
        return globals()[name]
    for mod in _LOADED_MODULES.values():
        val = _resolve_symbol_from_module(mod, name)
        if val is not _MISSING:
            globals()[name] = val
            if not name.startswith("_") and name not in __all__:
                __all__.append(name)
            return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
