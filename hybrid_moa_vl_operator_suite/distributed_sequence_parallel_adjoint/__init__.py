"""Subpackage for `distributed_sequence_parallel_adjoint`."""

from . import distributed_sequence_parallel_adjoint as _impl

__all__ = []
for _sym in getattr(_impl, "__all__", ()):
    if hasattr(_impl, _sym):
        globals()[_sym] = getattr(_impl, _sym)
        __all__.append(_sym)
