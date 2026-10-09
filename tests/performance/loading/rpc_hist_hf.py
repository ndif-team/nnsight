"""RPC size histogram for HF from_pretrained itself: mmap vs pread, 1 vs 4 workers, on Qwen3-1.7B -> GPU."""
import os, sys, time, gc, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rpc_hist import snapshot
import torch
from benchmark_loading import _patch_hf_workers, _patch_safetensors_backend, evict_model_pages, unload_model
from nnsight import TransformersModel
M = "Qwen/Qwen3-1.7B"
for backend, workers in [(None, 1), (None, 4), ("pread", 4)]:
    evict_model_pages(M); rw = _patch_hf_workers(workers); rb = _patch_safetensors_backend(backend)
    before = snapshot(); t0 = time.perf_counter()
    m = TransformersModel(M, task="text-generation", device_map="auto", dispatch=True, load_format="from_pretrained")
    torch.cuda.synchronize(); dt = time.perf_counter() - t0; after = snapshot()
    rb(); rw(); unload_model(m); gc.collect()
    d = after - before
    pages = {k[1]: v for k, v in d.items() if k[0] == "pages"}; infl = {k[1]: v for k, v in d.items() if k[0] == "inflight"}
    n = sum(pages.values()) or 1; avg = sum(p * c for p, c in pages.items()) * 4 / n / 1024
    print(f"\n== HF from_pretrained {backend or 'mmap'} workers={workers}: 3.8 GB in {dt:.1f}s = {3.8/dt:.2f} GB/s; {n} read RPCs, avg {avg:.2f} MB/RPC")
    print("   pages/RPC:", "  ".join(f"{p}p:{c}" for p, c in sorted(pages.items()) if c))
    print("   in flight:", "  ".join(f"{k}:{c}" for k, c in sorted(infl.items()) if c))
