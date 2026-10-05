"""Import compatibility for the tools venv (.venv-tools) on this machine.

A machine-wide application-control policy (Smart App Control) blocks
individual compiled files by reputation verdict; observed on this machine:
llvmlite.dll earlier, numba's experimental jitclass _box extension later.
A partially failed numba import leaves broken state, so this helper tests a
full real numba import first and, on any failure, purges every partially
imported numba/llvmlite module and installs a pass-through stub that keeps
librosa's pure-Python paths working. It neither loads blocked DLLs nor
changes any security setting.
"""

from __future__ import annotations

import sys
import types


def _try_full_numba_import() -> bool:
    """Check whether the real numba imports completely (no DLL blocks)."""
    try:
        import numba  # noqa: F401
        import numba.experimental.jitclass  # noqa: F401
    except Exception:
        for name in list(sys.modules):
            if name == "numba" or name.startswith("numba."):
                sys.modules.pop(name, None)
            elif name == "llvmlite" or name.startswith("llvmlite."):
                sys.modules.pop(name, None)
        return False
    return True


def install_tools_compat() -> None:
    """Install numba/resampy stubs unless real, working ones are importable."""
    import numpy as np

    if not _try_full_numba_import():
        numba = types.ModuleType("numba")

        def _passthrough_decorator(*_decorator_args, **_decorator_kwargs):
            def wrap(func):
                return func

            return wrap

        class _DtypeLike:
            def __getitem__(self, _key):
                return self

        numba.jit = _passthrough_decorator
        numba.stencil = _passthrough_decorator
        numba.vectorize = _passthrough_decorator
        numba.guvectorize = _passthrough_decorator
        numba.uint = _DtypeLike()
        numba.njit = _passthrough_decorator
        numba.cuda = types.SimpleNamespace(
            detect=lambda *args, **kwargs: None,
        )
        experimental = types.ModuleType("numba.experimental")
        experimental.jitclass = _passthrough_decorator
        numba.experimental = experimental
        sys.modules["numba"] = numba
        sys.modules["numba.experimental"] = experimental

    if "resampy" not in sys.modules:
        try:
            import resampy  # noqa: F401
        except Exception:
            resampy = types.ModuleType("resampy")
            resampy.__spec__ = None

            def resample(
                x,
                sr_orig,
                sr_new,
                *,
                filter="kaiser_best",
                axis=-1,
                **_kwargs,
            ):
                from math import gcd

                from scipy.signal import resample_poly

                x = np.asarray(x)
                divisor = gcd(int(sr_orig), int(sr_new))
                up, down = int(sr_new) // divisor, int(sr_orig) // divisor
                return resample_poly(x, up, down, axis=axis)

            resampy.resample = resample
            sys.modules["resampy"] = resampy
