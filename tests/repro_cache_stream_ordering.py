"""GPU repro harness for cache stream ordering on main.

Purpose:
- Verify whether Cache.add runs on an unexpected CUDA stream.
- Check if cached tensors can be consumed safely without torch.cuda.synchronize.

How to run:
	python tests/repro_cache_stream_ordering.py --runs 50

Notes:
- This script is intentionally standalone (not a pytest test) so it can be
  shared directly in PR comments as a reproducibility harness.
- It exits non-zero when it finds a real mismatch/failure signal.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

import nnsight
from nnsight.intervention.tracing.tracer import Cache


@dataclass
class AddRecord:
	module_path: str
	key: str
	stream_ptr: Optional[int]
	tensor_device: Optional[str]


def first_tensor(value: Any) -> Optional[torch.Tensor]:
	if isinstance(value, torch.Tensor):
		return value
	if isinstance(value, (list, tuple)):
		for item in value:
			t = first_tensor(item)
			if t is not None:
				return t
		return None
	if isinstance(value, dict):
		for item in value.values():
			t = first_tensor(item)
			if t is not None:
				return t
		return None
	return None


def run_repro(runs: int, model_id: str, prompt: str) -> int:
	if not torch.cuda.is_available():
		print("CUDA is not available in this environment. Cannot run GPU stream repro.")
		return 2

	device = "cuda:0"
	torch.set_grad_enabled(False)

	print(f"Loading model: {model_id} on {device}")
	model = nnsight.LanguageModel(model_id, device_map=device, dispatch=True)

	records: List[AddRecord] = []
	orig_add = Cache.add

	def wrapped_add(self, module_path: str, key: str, value: Any):
		stream_ptr: Optional[int]
		if torch.cuda.is_available():
			stream_ptr = int(torch.cuda.current_stream().cuda_stream)
		else:
			stream_ptr = None

		tensor = first_tensor(value)
		tensor_device = str(tensor.device) if tensor is not None else None
		records.append(
			AddRecord(
				module_path=module_path,
				key=key,
				stream_ptr=stream_ptr,
				tensor_device=tensor_device,
			)
		)

		return orig_add(self, module_path, key, value)

	Cache.add = wrapped_add

	mismatches = 0
	cross_stream_mismatches = 0
	nan_failures = 0

	consumer_stream = torch.cuda.Stream(device=torch.device(device))

	try:
		for i in range(runs):
			with model.trace(prompt) as tracer:
				# Force real GPU work in the cache transform by changing dtype.
				cache = tracer.cache(
					device=torch.device(device),
					dtype=torch.float32,
					modules=[model.transformer.h[0]],
				)

			cached_output = cache["model.transformer.h.0"].output

			# Same-stream consume (this is the natural path in user code).
			same_stream_value = cached_output.sum().item()

			# Different-stream consume (stress case). We intentionally do not
			# add any explicit stream wait here.
			with torch.cuda.stream(consumer_stream):
				cross_stream_tensor = cached_output.sum()

			consumer_stream.synchronize()
			cross_stream_value = cross_stream_tensor.item()

			# Fresh run baseline.
			with model.trace(prompt):
				fresh_output = model.transformer.h[0].output.save()
			fresh_value = fresh_output.to(torch.float32).sum().item()

			if math.isnan(same_stream_value) or math.isnan(cross_stream_value):
				nan_failures += 1

			if not math.isclose(same_stream_value, fresh_value, rel_tol=0.0, abs_tol=1e-4):
				mismatches += 1

			if not math.isclose(cross_stream_value, fresh_value, rel_tol=0.0, abs_tol=1e-4):
				cross_stream_mismatches += 1

			if (i + 1) % 10 == 0 or i == runs - 1:
				print(
					f"run {i + 1}/{runs} | same_stream_mismatch={mismatches} "
					f"cross_stream_mismatch={cross_stream_mismatches} nan_failures={nan_failures}"
				)

	finally:
		Cache.add = orig_add

	stream_ptrs = sorted({r.stream_ptr for r in records if r.stream_ptr is not None})
	null_stream_seen = any(ptr == 0 for ptr in stream_ptrs)
	cuda_records = [r for r in records if r.tensor_device and "cuda" in r.tensor_device]

	print("\n=== Summary ===")
	print(f"cache_add_calls={len(records)}")
	print(f"cuda_cache_add_calls={len(cuda_records)}")
	print(f"unique_cache_add_stream_ptrs={stream_ptrs}")
	print(f"null_stream_seen={null_stream_seen}")
	print(f"same_stream_mismatches={mismatches}")
	print(f"cross_stream_mismatches={cross_stream_mismatches}")
	print(f"nan_failures={nan_failures}")

	# Fail only when there is an actual numerical mismatch/failure signal.
	if mismatches > 0 or cross_stream_mismatches > 0 or nan_failures > 0:
		print("RESULT: FAIL (found mismatch/failure signal)")
		return 1

	print("RESULT: PASS (no mismatch signal without global synchronize)")
	return 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Repro harness for cache stream ordering")
	parser.add_argument("--runs", type=int, default=50, help="Number of repeated trials")
	parser.add_argument("--model", type=str, default="openai-community/gpt2")
	parser.add_argument(
		"--prompt",
		type=str,
		default="Madison Square Garden is located in the city of",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	sys.exit(run_repro(args.runs, args.model, args.prompt))
