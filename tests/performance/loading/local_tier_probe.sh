#!/bin/bash
# Local-tier probe: is /local writable and persistent, and how fast are /tmp (NVMe) and /dev/shm reads.
SRC=$(readlink -f /work/nvme/bdnh/zwang83/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/model-00001-of-00002.safetensors)
echo "-- /local: $(ls -ld /local | cut -c1-40); writable: $(touch /local/.probe_$USER 2>&1 && echo yes || echo no)"; ls /local | head -5
for tier in /tmp /dev/shm; do
  f=$tier/probe_shard.safetensors
  t0=$(date +%s.%N); cp "$SRC" "$f"; t1=$(date +%s.%N)
  sz=$(stat -c %s "$f"); echo "-- $tier: copy $((sz/1024/1024)) MB from Lustre in $(echo "$t1 - $t0" | bc | cut -c1-5)s"
  python3 - "$f" <<'PY'
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
f = sys.argv[1]; sz = os.path.getsize(f); n_chunks = 64; chunk = (sz + n_chunks - 1) // n_chunks
def evict():
    fd = os.open(f, os.O_RDONLY); os.fsync(fd) if False else None; os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def rd(i):
    with open(f, "rb", buffering=0) as h:
        h.seek(i * chunk); left = chunk
        while left > 0:
            b = h.read(min(16 << 20, left));
            if not b: break
            left -= len(b)
for threads in (1, 4, 16):
    evict(); t = time.perf_counter()
    with ThreadPoolExecutor(threads) as p: list(p.map(rd, range(n_chunks)))
    dt = time.perf_counter() - t; print(f"   read() threads={threads:<3d} {sz/2**30/dt:6.2f} GB/s  ({'cold' if 'shm' not in f else 'RAM'})")
PY
  rm -f "$f"
done
