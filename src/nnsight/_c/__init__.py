"""Low-level, process-wide helpers nnsight installs at import.

- ``py_mount`` — a small C extension that writes a method into ``object``'s
  type dict (bypassing its read-only ``mappingproxy``), so it appears on *all*
  objects. nnsight uses it to add ``.save()`` universally. Optional: if it didn't
  build, ``mount``/``unmount`` are ``None`` and callers fall back to
  ``nnsight.save(value)``.
- [`backtrace`][nnsight._c.backtrace] — neutralizes glibc ``backtrace()`` so a torch C++ error
  raised inside an interleaving greenlet surfaces as a normal Python exception
  instead of segfaulting the process (see [`backtrace.install`][nnsight._c.backtrace.install]).
"""

from .backtrace import install as install_backtrace_guard  # noqa: F401

try:
    from .py_mount import mount, unmount  # noqa: F401
except Exception:  # noqa: BLE001 — optional extension; callers fall back to save()
    mount = None
    unmount = None

__all__ = ["mount", "unmount", "install_backtrace_guard"]
