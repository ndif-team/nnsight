"""Footgun-containment seccomp filter for the isolated GPU worker (x86-64).

Per the agreed threat model (contain *mistakes*, not a determined adversary): after
CUDA + torch are warmed and the intervention is deserialized, the worker calls
:func:`lock_down`, which installs a minimal seccomp-BPF filter making new
``open``/``openat`` (filesystem), ``socket``/``connect`` (network), and ``execve``
(spawn) syscalls fail with EPERM. CUDA keeps working because it talks to the
already-open ``/dev/nvidia*`` fds via ioctl/mmap, and the control Pipe + CUDA-IPC
buffer use already-open fds (read/write/ioctl), not new opens.

No external deps: the BPF program is assembled by hand and installed via
``prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ...)``. Proven in
prototypes/mediator-sandbox/gpu_sandbox (test_safety: 9/9 contained).
"""
import ctypes
import struct

# x86-64 syscall numbers
_NR = {
    "open": 2, "openat": 257, "openat2": 437,
    "socket": 41, "connect": 42, "execve": 59, "execveat": 322,
}
_AUDIT_ARCH_X86_64 = 0xC000003E
_RET_KILL_PROCESS = 0x80000000
_RET_ERRNO = 0x00050000
_RET_ALLOW = 0x7FFF0000
_EPERM = 1
# BPF opcodes
_LD_W_ABS = 0x20
_JMP_JEQ_K = 0x15
_RET_K = 0x06
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2


def _build_filter(blocked):
    instrs = [
        (_LD_W_ABS, 0, 0, 4),                      # A = arch (seccomp_data offset 4)
        (_JMP_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),    # if x86-64: skip the kill
        (_RET_K, 0, 0, _RET_KILL_PROCESS),         # else kill (block arch-bypass)
        (_LD_W_ABS, 0, 0, 0),                      # A = syscall nr (offset 0)
    ]
    for nr in blocked:
        instrs.append((_JMP_JEQ_K, 0, 1, nr))      # if A == nr: next else skip next
        instrs.append((_RET_K, 0, 0, _RET_ERRNO | _EPERM))
    instrs.append((_RET_K, 0, 0, _RET_ALLOW))      # default: allow
    return instrs


class _sock_fprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def lock_down(block_fs: bool = True, block_net: bool = True) -> None:
    """Install the seccomp filter. Call AFTER torch/CUDA are warmed + the
    intervention is deserialized (deserialization may open files)."""
    blocked = []
    if block_fs:
        blocked += [_NR["open"], _NR["openat"], _NR["openat2"]]
    if block_net:
        blocked += [_NR["socket"], _NR["connect"]]
    blocked += [_NR["execve"], _NR["execveat"]]    # no spawning new programs either
    instrs = _build_filter(blocked)

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")

    prog = b"".join(struct.pack("HBBI", *i) for i in instrs)
    buf = ctypes.create_string_buffer(prog, len(prog))
    fprog = _sock_fprog(len(instrs), ctypes.cast(buf, ctypes.c_void_p))
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP failed")
