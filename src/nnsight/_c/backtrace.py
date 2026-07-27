"""Stop a torch C++ error from segfaulting the process when it is raised inside
an interleaving greenlet.

nnsight runs intervention code in greenlets (see
:mod:`nnsight.intervention.interleaver`). greenlet time-shares one OS thread
stack by copying stack *slices* in and out of the heap on every switch. When a
torch op raises a ``c10`` error while a worker greenlet is running, torch's
``c10::Error`` constructor eagerly captures a C++ backtrace via glibc
``backtrace()`` -- which walks the *raw machine stack*. It walks the worker's
live frames fine, reaches greenlet's switch trampoline (which has no clean DWARF
unwind info), and then continues into the shared-stack region below, now holding
stale bytes from another worker's saved/restored slice. libgcc's unwinder
computes a garbage frame there and SIGSEGVs -- so a plain shape error in a user's
intervention becomes a hard process crash instead of a normal Python exception.

The one fragile operation is that ``backtrace()`` call. This module removes it:
it overwrites glibc ``backtrace()`` in-process so it returns 0 (no frames)
without walking the stack. torch then builds the error with an empty C++
backtrace and it propagates as an ordinary Python exception. Python tracebacks
and error messages are unaffected; only torch's (rarely used) *C++* backtrace
string is emptied. This is the in-process equivalent of ``LD_PRELOAD``-ing a
no-op ``backtrace()``.

Gated by ``CONFIG.APP.DISABLE_CPP_BACKTRACE`` (env
``NNSIGHT_DISABLE_CPP_BACKTRACE``). Only glibc on x86-64 Linux is patched -- that
is where the crash lives; musl's ``backtrace()`` is already a no-op stub and
other platforms use different unwinders, so they are left untouched. Every step
fails safe: if anything is unexpected the process is left exactly as it was.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys

# x86-64: `xor eax, eax ; ret` -- makes backtrace(void**, int) return 0 and leave
# the caller's buffer untouched, so torch captures an empty C++ backtrace.
_RET_ZERO = b"\x31\xc0\xc3"

_installed = False


def install() -> bool:
    """Neutralize glibc ``backtrace()`` so a c10 error on a greenlet can't crash.

    Returns ``True`` if the guard is in place (including on a repeat call), and
    ``False`` if it was skipped -- unsupported platform, no glibc ``backtrace``,
    or a page could not be made writable. Idempotent and never raises.
    """
    global _installed
    if _installed:
        return True

    # The crash is specifically glibc backtrace() + libgcc's DWARF unwinder on
    # x86-64. Elsewhere the ingredients differ (musl backtrace() is a no-op stub;
    # other arches/unwinders don't hit this), so patch nothing.
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return False

    try:
        libc = ctypes.CDLL(None)  # already-loaded C library (RTLD_DEFAULT lookup)
        if not hasattr(libc, "backtrace"):
            return False
        addr = ctypes.cast(libc.backtrace, ctypes.c_void_p).value
        if not addr:
            return False

        mprotect = libc.mprotect
        mprotect.restype = ctypes.c_int
        mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

        page = os.sysconf("SC_PAGESIZE")
        start = addr & ~(page - 1)
        length = ((addr + len(_RET_ZERO) - start) + page - 1) & ~(page - 1)

        prot_read, prot_write, prot_exec = 0x1, 0x2, 0x4
        if mprotect(start, length, prot_read | prot_write | prot_exec) != 0:
            return False
        # x86 keeps instruction and data caches coherent, so the overwrite takes
        # effect with no explicit icache flush.
        ctypes.memmove(addr, _RET_ZERO, len(_RET_ZERO))
        mprotect(start, length, prot_read | prot_exec)  # best-effort re-lock

        _installed = True
        return True
    except Exception:
        return False
