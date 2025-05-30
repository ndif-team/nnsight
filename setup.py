from setuptools import setup, Extension

extensions = [
    Extension(
        "nnsight._c.py_mount",
        ["src/nnsight/_c/py_mount.c"],
    )
]

setup(
    ext_modules=extensions,
)