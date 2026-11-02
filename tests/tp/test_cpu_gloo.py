"""Tensor parallelism on CPU, so CI covers it at all.

`test_sharded_tracing.py` is the same comparison on GPUs and is skipped
everywhere without two of them — which is why transformers 5.16 could remove the
ground the TP code stands on and no test said so. torch's DeviceMesh runs over
gloo on CPU, and transformers' TP init has a CPU branch, so the whole path —
sharded weights, the style's transforms, nnsight's gather and re-split — runs in
a couple of seconds on any machine. It is the same worker as the GPU tests,
launched with ``--device cpu``.

What CPU cannot cover is real kernels and dtypes: this proves the *layout* logic,
not that a bf16 model on eight cards gives the right numbers. `tests/tp/` still
owns that.

Two things are checked, and they fail differently:

* **nnsight noticed the model is sharded.** A detection regression — transformers
  moving its plan off the modules, say — makes fragments inert, and then every
  sharded value is silently handed over as one rank's slice. The trace still runs
  and the numbers are quietly wrong, so nothing but an explicit assertion catches
  it. That is what happened on 5.16.
* **rank 0 matches the single-process run**, within a tolerance, because an
  all-reduce sums in a different order than one matmul. A mismatch of order 1 is
  a layout error; 1e-4 is float arithmetic.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest
import torch

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")


def _nnsight_path() -> str:
    """The directory to put on the worker's ``PYTHONPATH`` so it imports the same
    nnsight this session did.

    Not the repo's ``src`` unconditionally: a non-editable install (which is what
    CI does) builds the ``_c`` extensions into site-packages and leaves the source
    tree without them, so forcing ``src`` ahead of site-packages would import a
    half-built package and drop ``.save()`` off every tensor. Following the
    already-imported module keeps an editable checkout and an installed wheel both
    working, and tests whichever one the session is actually exercising.
    """
    import nnsight

    return os.path.dirname(os.path.dirname(os.path.abspath(nnsight.__file__)))


REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TP_SIZE = 2

# REPO with a vocabulary TP_SIZE does not divide, so the last rank holds the
# shorter slice of the embedding and of `lm_head`'s output. Tied as well, as on
# Llama-3.2, so the two share one vocab-parallel weight; REPO is untied.
UNEVEN_VOCAB = 32001

# float32 on a 2-rank CPU mesh drifts far less than bf16 on 4 GPUs, so this is
# tighter than the GPU suite's 1e-3.
DRIFT = 1e-4


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run(tp: int, out: str, repo: str) -> None:
    """Run the worker at ``tp`` CPU ranks, writing one file per rank into ``out``."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [_nnsight_path(), os.environ.get("PYTHONPATH")])
        ),
        # The empty string is what makes this a CPU run: transformers picks the
        # mesh device from `torch._C._get_accelerator()`, which reports CPU only
        # when no CUDA device is visible. Without this the ranks would each try
        # to take a card and the run would need the hardware after all.
        "CUDA_VISIBLE_DEVICES": "",
        # gloo spawns a thread per rank per collective; the default thread count
        # oversubscribes a shared machine for no gain on a model this size.
        "OMP_NUM_THREADS": "1",
    }
    command = [sys.executable]
    if tp > 1:
        command += [
            "-m", "torch.distributed.run",
            f"--nproc_per_node={tp}",
            f"--master_port={_free_port()}",
        ]
    command += [WORKER, "--tp", str(tp), "--repo", repo, "--out", out, "--device", "cpu"]

    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            f"tp={tp} CPU worker failed ({completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr ---\n{completed.stderr[-4000:]}"
        )


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """max|a-b| / max|b| — scale-free, so it reads the same at any layer."""
    if actual.shape != expected.shape:
        return float("inf")
    scale = expected.abs().max().item()
    return (actual - expected).abs().max().item() / (scale if scale else 1.0)


def _uneven_vocab_repo(tmp_path_factory) -> str:
    """REPO's shape with `UNEVEN_VOCAB` and tied embeddings, saved by the running
    transformers."""
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    config = AutoConfig.from_pretrained(REPO)
    config.vocab_size = UNEVEN_VOCAB
    config.tie_word_embeddings = True
    config.pad_token_id = None  # REPO's -1 fails the generation config check on save
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("llama_uneven_vocab")
    LlamaForCausalLM(config).save_pretrained(path)
    AutoTokenizer.from_pretrained(REPO).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module", params=["hub", "uneven_vocab"])
def runs(request, tmp_path_factory) -> tuple[dict, list[dict]]:
    """``(reference, [per-rank sharded results])`` from two subprocess runs."""
    repo = REPO
    if request.param == "uneven_vocab":
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__).release < (5, 21):
            pytest.skip("transformers < 5.21 refuses a vocabulary tp_size does not divide")
        repo = _uneven_vocab_repo(tmp_path_factory)

    reference_dir = tmp_path_factory.mktemp("cpu_tp1")
    sharded_dir = tmp_path_factory.mktemp(f"cpu_tp{TP_SIZE}")

    _run(1, str(reference_dir), repo)
    _run(TP_SIZE, str(sharded_dir), repo)

    reference = torch.load(os.path.join(reference_dir, "rank0.pt"), weights_only=False)
    sharded = [
        torch.load(os.path.join(sharded_dir, f"rank{rank}.pt"), weights_only=False)
        for rank in range(TP_SIZE)
    ]
    return reference, sharded


def test_the_sharded_run_produced_every_value(runs) -> None:
    """The worker completed on every rank — nothing deadlocked or diverged."""
    reference, sharded = runs
    for rank, result in enumerate(sharded):
        assert set(result) == set(reference), f"rank {rank} recorded a different set"


def test_the_ranks_agree(runs) -> None:
    """Every rank saw the same whole tensors — bit for bit.

    A difference here is the deadlock/corruption class: the ranks disagreed about
    what the value was, which means the collectives did not line up.
    """
    _, sharded = runs
    first, *rest = sharded
    for rank, result in enumerate(rest, start=1):
        for name, value in first.items():
            assert torch.equal(value, result[name]), f"rank {rank} disagrees on {name}"


@pytest.mark.parametrize(
    "name",
    [
        "gate_proj_out",      # a column-parallel output: this rank's slice
        "down_proj_in",       # a row-parallel input: pre-split
        "layer_out",
        "baseline_logits",
        "nograd_logits",        # a plain local tensor: the rule's width decides
        "partial_edit_logits",  # an edit straddling the rank boundary
        "embed_out",            # a vocab-parallel embedding's single-use mask
        "embed_edited_logits",  # ... edited and handed back
        "adhoc_colwise",
        "adhoc_rowwise",
        "adhoc_hooked",   # hook=True must not skip the bracket
        "adhoc_nested",   # an ad-hoc call inside its own open handoff
        "skip_read_back",   # a .skip replacement is not gathered
        "skip_logits",
        "adhoc_lens",
        "cached_gate_out",
        "partial_backward_grad",   # backward through a row-parallel output
        "source_colwise_gathered",   # tp.gather on an op inside a sharded module
        "manual_gathered_heads",   # tp.gather on a value between two shards
        "manual_ablated_logits",   # ... edited and tp.shard-ed back
        "batched_first",
        "batched_edited_logits",
        "generated_steps",
    ],
)
def test_rank0_matches_the_single_process_run(runs, name) -> None:
    """A sharded value arrives whole, and matches what one process computed."""
    reference, sharded = runs
    drift = _rel(sharded[0][name], reference[name])
    assert drift < DRIFT, (
        f"{name}: relative error {drift:.2e} against the 1-rank run "
        f"(shapes {tuple(sharded[0][name].shape)} vs {tuple(reference[name].shape)}). "
        "Order 1 means the gather or the re-split is wrong; this is not drift."
    )


def test_generation_is_identical(runs) -> None:
    """Greedy decoding produced the same token ids, not merely close logits."""
    reference, sharded = runs
    assert torch.equal(sharded[0]["generated"], reference["generated"])
