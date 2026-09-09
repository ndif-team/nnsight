"""Build the optional C extension.

Package metadata lives in pyproject.toml; this only adds the ``nnsight._c.py_mount``
extension (which mounts ``.save()`` on all objects). It's marked ``optional`` so a
failed build is a warning, not a hard error — nnsight then falls back to
``nnsight.save(value)``.
"""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "nnsight._c.py_mount",
            ["src/nnsight/_c/py_mount.c"],
            optional=True,
        )
    ],
)
