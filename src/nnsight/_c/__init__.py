"""Compiled helper for mounting methods on every Python object.

:mod:`py_mount` is a small C extension that writes a method into ``object``'s
type dict (bypassing its read-only ``mappingproxy``), so it appears on *all*
objects. nnsight uses it to add ``.save()`` universally. It's an optional
extension — if it didn't build, importing this fails and callers fall back to
``nnsight.save(value)``.
"""

from .py_mount import mount, unmount  # noqa: F401

__all__ = ["mount", "unmount"]
