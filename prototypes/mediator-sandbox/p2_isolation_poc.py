#!/usr/bin/env python3
"""P2 (host-level sibling sandbox) isolation PoC.

Runs the real ndif escape-suite gadgets (test_protector_escapes.py) as *user
code inside a host-level bubblewrap jail* and shows each is INERT — the gadget
still runs (P2 has NO import/builtin whitelist; that's the point), but it cannot
read a host file, write a host file, or reach the network, and it sees no
foreign processes.

This is the inverse of the in-process Protector: the Protector tried to *forbid*
dangerous APIs (os, ctypes, subprocess, the __subclasses__ walk) and failed on
all 10. P2 *allows* them and removes everything they could touch.

Design ref: docs/developing/mediator-isolation-sandbox.md (§6.5 P2, §6.1 jail).
Mechanism verified: a socketpair fd inherited into a net=none CPU-only jail.

Run with the env interpreter so sys.executable points into the bound env:
    /disk/u/zikai/anaconda3/bin/python prototypes/mediator-sandbox/p2_isolation_poc.py
"""
import json
import os
import shutil
import socket
import subprocess
import sys

PYBIN = sys.executable                 # the (untrusted) interpreter run inside the jail
ENV_ROOT = os.path.dirname(os.path.dirname(PYBIN))  # e.g. /disk/u/zikai/anaconda3 — bound ro
HOST_DIR = os.path.expanduser("~/.p2_poc")          # host-only; deliberately NOT bound into jail
SECRET = os.path.join(HOST_DIR, "secret.txt")
PWNED = os.path.join(HOST_DIR, "pwned")
SECRET_CONTENT = "TOPSECRET-MEDIATOR-ACTIVATIONS-9f3a"

# Worker entrypoint that runs INSIDE the jail: rebuild the socket from the
# inherited fd, recv one gadget, exec it (no whitelist), report the verdict it
# set. Modules are preimported to mirror how an nnsight closure arrives with a
# populated globals — so the __subclasses__ / __globals__ walks find their targets.
WORKER = r'''
import os, sys, socket, json
import subprocess, pickle, inspect, types
for _m in ("numpy", "urllib3"):
    try: __import__(_m)
    except Exception: pass
fd = int(os.environ["WORKER_FD"])
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=fd)
buf = b""
while not buf.endswith(b"\n"):
    chunk = s.recv(65536)
    if not chunk: break
    buf += chunk
msg = json.loads(buf.decode())
ns = dict(globals()); ns["verdict"] = "ran"; ns["detail"] = ""
try:
    exec(msg["code"], ns)
    v, d = ns.get("verdict", "ran"), str(ns.get("detail", ""))[:200]
except Exception as e:
    v, d = "EXC", (type(e).__name__ + ": " + str(e))[:200]
s.sendall((json.dumps({"verdict": v, "detail": d}) + "\n").encode())
s.close()
'''


def run_in_jail(code: str, timeout: int = 60) -> dict:
    """Spawn one bwrap jail, hand it a gadget over an inherited socket, get the verdict."""
    broker, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    os.set_inheritable(worker.fileno(), True)
    cmd = [
        "bwrap", "--unshare-all",                       # userns+net+pid+ipc+uts+mount, fresh
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", ENV_ROOT, ENV_ROOT,                # the python env, read-only
        "--proc", "/proc",                              # fresh proc (real PID isolation — host can)
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--die-with-parent",
        "--setenv", "CUDA_VISIBLE_DEVICES", "",         # no GPU
        "--setenv", "WORKER_FD", str(worker.fileno()),
        PYBIN, "-c", WORKER,
    ]
    p = subprocess.Popen(cmd, pass_fds=[worker.fileno()])
    worker.close()
    broker.sendall((json.dumps({"code": code}) + "\n").encode())
    broker.settimeout(timeout)
    data = b""
    try:
        while not data.endswith(b"\n"):
            chunk = broker.recv(65536)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        data = b'{"verdict": "TIMEOUT", "detail": "no reply"}\n'
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    broker.close()
    try:
        return json.loads(data.decode())
    except Exception:
        return {"verdict": "PARSE_ERR", "detail": data[:120].decode(errors="replace")}


# Each gadget is the real escape primitive from the suite, then it attempts the
# *host effect* and self-reports. __SECRET__/__PWNED__ are substituted below.
GADGETS = [
    ("escape1+9 subclasses→Popen→read host secret", r'''
Popen = None
for c in ().__class__.__mro__[1].__subclasses__():
    if c.__name__ == "Popen" and c.__module__ == "subprocess":
        Popen = c; break
if Popen is None:
    verdict, detail = "N/A", "Popen not in subclass walk"
else:
    try:
        pr = Popen(["cat", "__SECRET__"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        o, e = pr.communicate(timeout=5)
        verdict = "LEAKED" if b"TOPSECRET" in o else "CONTAINED"
        detail = (o[:40] if o else e[:80]).decode(errors="replace")
    except Exception as ex:
        verdict, detail = "CONTAINED", "%s: %s" % (type(ex).__name__, ex)
'''),
    ("escape2 format-string __globals__→os→system(touch host)", r'''
os_mod = subprocess.Popen.__init__.__globals__["os"]
proof = "{0.__globals__[os]}".format(subprocess.Popen.__init__)
rc = os_mod.system("touch __PWNED__ 2>/dev/null")
verdict, detail = "ran", "reached %s ; os.system rc=%s" % ("os" in proof, rc)
'''),
    ("escape3 numpy.ctypeslib→libc.open(host secret)", r'''
import numpy
ct = numpy.ctypeslib.ctypes
libc = ct.CDLL("libc.so.6")
fd = libc.open(b"__SECRET__", 0)            # O_RDONLY
if fd >= 0:
    b = ct.create_string_buffer(64); n = libc.read(fd, b, 64); libc.close(fd)
    verdict = "LEAKED" if b"TOPSECRET" in b.raw else "CONTAINED"
    detail = "fd=%s n=%s" % (fd, n)
else:
    verdict, detail = "CONTAINED", "libc.open fd=%s (host file absent in jail)" % fd
'''),
    ("escape7 urllib3 egress to 1.1.1.1", r'''
try:
    import urllib3
    pool = urllib3.PoolManager(retries=False, timeout=3.0)
    r = pool.request("GET", "http://1.1.1.1/")
    verdict, detail = "LEAKED", "egress ok status=%s" % r.status
except Exception as ex:
    verdict, detail = "CONTAINED", "%s: %s" % (type(ex).__name__, str(ex)[:90])
'''),
    ("escape9 end-to-end RCE: Popen.globals→os.system(touch host)", r'''
Popen = None
for c in ().__class__.__mro__[1].__subclasses__():
    if c.__name__ == "Popen" and c.__module__ == "subprocess":
        Popen = c; break
os_mod = Popen.__init__.__globals__["os"]
rc = os_mod.system("touch __PWNED__ 2>/dev/null")
verdict, detail = "ran", "os.system rc=%s (host effect checked by orchestrator)" % rc
'''),
    ("escape10 pickle __reduce__ fired DURING deserialize-in-jail", r'''
class G:
    def __reduce__(self):
        return (os.system, ("touch __PWNED__ 2>/dev/null",))
payload = pickle.dumps(G())
pickle.loads(payload)                       # P2: untrusted unpickle happens inside the jail
verdict, detail = "ran", "pickle gadget fired inside jail"
'''),
    ("PID-namespace isolation (validates §6.3 fixed at host level)", r'''
pids = sorted(int(x) for x in os.listdir("/proc") if x.isdigit())
verdict = "ISOLATED" if len(pids) <= 3 else "LEAK"
detail = "visible pids=%s" % pids
'''),
]


def main():
    if shutil.which("bwrap") is None:
        sys.exit("bwrap not found on host — P2 jail unavailable")
    os.makedirs(HOST_DIR, exist_ok=True)
    with open(SECRET, "w") as f:
        f.write(SECRET_CONTENT)
    if os.path.exists(PWNED):
        os.remove(PWNED)

    print("P2 isolation PoC — gadgets run inside a host-level bwrap jail")
    print("  jail: %s (CPU-only, net=none, ro-fs allowlist, fresh PID ns)" % PYBIN)
    print("  host secret (NOT bound into jail): %s" % SECRET)
    print("-" * 78)

    leaked = []
    for name, tmpl in GADGETS:
        code = tmpl.replace("__SECRET__", SECRET).replace("__PWNED__", PWNED)
        res = run_in_jail(code)
        v, d = res.get("verdict", "?"), res.get("detail", "")
        bad = v in ("LEAKED", "LEAK", "TIMEOUT", "PARSE_ERR")
        if bad:
            leaked.append(name)
        print("  [%s] %s" % ("LEAK" if bad else "ok  ", name))
        print("        verdict=%s  %s" % (v, d))

    # Host-side ground truth: the gadgets must have touched nothing on the host.
    print("-" * 78)
    host_pwned = os.path.exists(PWNED)
    print("  host-side check: PWNED file created on host? %s" % ("YES (LEAK)" if host_pwned else "no"))
    print("  host-side check: secret content ever returned? %s"
          % ("YES (LEAK)" if any("TOPSECRET" in g for g in leaked) else "no"))

    ok = (not leaked) and (not host_pwned)
    print("=" * 78)
    print("RESULT: %s" % ("PASS — every gadget ran but was inert" if ok
                          else "FAIL — leaks: %s%s" % (leaked, " +host-write" if host_pwned else "")))

    # cleanup
    shutil.rmtree(HOST_DIR, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
