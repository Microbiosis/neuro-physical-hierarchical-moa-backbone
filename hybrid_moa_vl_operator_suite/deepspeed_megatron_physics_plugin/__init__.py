"""Subpackage for `deepspeed_megatron_physics_plugin`."""

from . import deepspeed_megatron_physics_plugin as _impl

__all__ = []
for _sym in getattr(_impl, "__all__", ()):
    if hasattr(_impl, _sym):
        globals()[_sym] = getattr(_impl, _sym)
        __all__.append(_sym)
