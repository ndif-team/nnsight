"""Per-read-method Lustre RPC size histogram: snapshot osc.*.rpc_stats before and
after reading one cold shard, and print the delta of the READ 'pages per rpc' rows."""
import glob, mmap, os, re, subprocess, sys, time, collections
import torch
torch.set_num_threads(1)
hub = "/work/nvme/bdnh/zwang83/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/"
path = os.path.realpath(sorted(glob.glob(hub + "*.safetensors"))[0])
SIZE = os.path.getsize(path)

def snapshot():
    out = subprocess.run(["lctl", "get_param", "osc.dltawork-*.rpc_stats"], capture_output=True, text=True).stdout
    hist = collections.Counter(); ra = None
    section = None
    for line in out.splitlines():
        if line.startswith("pages per rpc"): section = "pages"; continue
        if line.startswith("rpcs in flight"): section = "inflight"; continue
        if line.startswith("offset"): section = None; continue
        m = re.match(r"\s*(\d+):\s+(\d+)\s+\d+\s+\d+\s+\|\s+(\d+)", line)
        if m and section:
            hist[(section, int(m.group(1)))] += int(m.group(3)) if False else int(m.group(2))  # read column
    return hist

def evict():
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)

def rd():
    with open(path, "rb", buffering=0) as f:
        while f.read(16 << 20): pass
def mm(threads):
    torch.set_num_threads(threads)
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    v = torch.frombuffer(m, dtype=torch.uint8)
    for off in range(0, v.numel(), 64 << 20): v[off:off + (64 << 20)].clone()
    del v; m.close()

def run_all():
  for name, fn in [("read() 16MB", rd), ("mmap sequential (1 copy thread)", lambda: mm(1)), ("mmap scattered (16 copy threads)", lambda: mm(16))]:
      evict(); before = snapshot(); t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0; after = snapshot()
      d = after - before
      pages = {k[1]: v for k, v in d.items() if k[0] == "pages"}
      infl = {k[1]: v for k, v in d.items() if k[0] == "inflight"}
      n = sum(pages.values()) or 1
      avg_kb = sum(p * c for p, c in pages.items()) * 4 / n
      print(f"\n== {name}: {SIZE/2**30:.2f} GiB in {dt:.1f}s = {SIZE/2**30/dt:.2f} GB/s; {n} read RPCs, avg {avg_kb/1024:.2f} MB/RPC")
      print("   pages/RPC histogram (read):", "  ".join(f"{p}p:{c}" for p, c in sorted(pages.items()) if c))
      print("   RPCs in flight when issued:", "  ".join(f"{k}:{c}" for k, c in sorted(infl.items()) if c))


if __name__ == "__main__":
    run_all()
