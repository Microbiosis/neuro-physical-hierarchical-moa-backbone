"""
Interdisciplinary Hybrid MoA Operator Suite (`hybrid_moa_vl_operator_suite`).

Organizes each standalone operator into its own subpackage directory alongside grouped
families (`payload/`, `spin_chirality/`) and the 5D verification suite (`audit/`);
there is no orchestration layer - every module is independently importable and carries
its own verification suite.

Zero-hardcoded-table dynamic package initializer:
1. Recursively discovers all leaf `.py` operator/kernel modules across subpackages via
   `pkgutil.iter_modules` (ordering `moa_*` first and `audit.*` verification modules last).
2. Dynamically scans each module's `__all__` (or inspects objects defined in `mod.__name__`)
   and resolves every symbol via `getattr` with fault-tolerant fallback to nested `*Kernels`,
   `*Simulator`, and `*Engine` classes.
3. Strictly switches to grouped subpackage paths (no legacy flat leaf module aliases).
4. Provides a PEP 562 module-level `__getattr__` hook for on-demand cross-submodule lookup.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from types import ModuleType
from typing import Dict, List, Tuple

_MISSING = object()
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
_OPTIONAL_EXTERNAL_DEPS = {"torch", "triton", "scipy"}
_CLI_ONLY_SYMBOLS = {"main"}
_LOADED_MODULES: Dict[str, ModuleType] = {}
__all__: List[str] = []


def _discover_leaf_modules() -> List[Tuple[str, str, str]]:
    """Discover `(subpkg_name, leaf_name, rel_mod_path)` entries without a hardcoded list."""
    entries: List[Tuple[str, str, str]] = []
    for pkg_info in pkgutil.iter_modules([_PKG_DIR]):
        if pkg_info.name.startswith("_"):
            continue
        if pkg_info.ispkg:
            subpkg_dir = os.path.join(_PKG_DIR, pkg_info.name)
            for mod_info in pkgutil.iter_modules([subpkg_dir]):
                if not mod_info.ispkg and not mod_info.name.startswith("_"):
                    entries.append(
                        (pkg_info.name, mod_info.name, f"{pkg_info.name}.{mod_info.name}")
                    )
        else:
            entries.append(("", pkg_info.name, pkg_info.name))

    def _sort_key(entry: Tuple[str, str, str]) -> tuple:
        subpkg_name, leaf_name, rel_path = entry
        if leaf_name.startswith("moa_"):
            return (0, rel_path)
        if leaf_name.endswith(("_operators", "_kernels")):
            return (1, rel_path)
        if "architecture" in leaf_name:
            return (2, rel_path)
        if subpkg_name == "audit":
            return (4, rel_path)
        return (3, rel_path)

    return sorted(entries, key=_sort_key)


def _load_submodule(rel_mod_path: str) -> ModuleType:
    if __package__:
        return importlib.import_module(f".{rel_mod_path}", __package__)
    return importlib.import_module(rel_mod_path)


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


def _is_optional_dep_missing(exc: ModuleNotFoundError) -> bool:
    top_name = (exc.name or "").split(".")[0]
    return top_name in _OPTIONAL_EXTERNAL_DEPS


def _bootstrap_package_namespace() -> None:
    g = globals()
    for subpkg_name, leaf_name, rel_mod_path in _discover_leaf_modules():
        try:
            mod = _load_submodule(rel_mod_path)
        except ModuleNotFoundError as exc:
            if _is_optional_dep_missing(exc):
                continue
            raise
        _LOADED_MODULES[rel_mod_path] = mod
        if subpkg_name:
            if subpkg_name not in g:
                g[subpkg_name] = _load_submodule(subpkg_name)
        else:
            g[leaf_name] = mod

        subpkg_mod = g.get(subpkg_name) if subpkg_name else None
        if subpkg_name == "spin_chirality" and subpkg_mod is not None:
            if leaf_name == "operator":
                mod_symbols = [
                    str(s)
                    for s in getattr(subpkg_mod, "__all__", ())
                    if isinstance(s, str)
                    and not s.startswith("_")
                    and not isinstance(getattr(subpkg_mod, s, None), ModuleType)
                ]
            elif leaf_name == "lattice4d":
                mod_symbols = _discover_module_symbols(mod)
            else:
                mod_symbols = []
            symbol_source = subpkg_mod if leaf_name == "operator" else mod
        else:
            mod_symbols = _discover_module_symbols(mod)
            symbol_source = mod

        for sym in mod_symbols:
            if sym in _CLI_ONLY_SYMBOLS:
                continue
            val = _resolve_symbol_from_module(symbol_source, sym)
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
    if name.startswith("_") or name in _CLI_ONLY_SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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
