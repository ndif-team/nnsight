"""Multi-node NNsight vLLM tests.

Validates that NNsight interventions work correctly when vLLM uses
Ray distributed executor with tensor parallelism across separate
nodes (or Docker containers simulating multi-node).

Run the Ray cluster first (see docker-compose.yml or README.md):
    docker compose up -d

Then run this script from the host:
    python test_multinode.py

Or with a custom Ray GCS address:
    RAY_ADDRESS=some-host:6379 python test_multinode.py

Clean up:
    docker compose down
"""

import os
import subprocess
import time
import traceback

# Set RAY_ADDRESS default before importing ray, so the EngineCore subprocess
# (which inherits our environment) also connects to the right cluster.
# Must be a GCS address (host:port), NOT a ray:// client address.
os.environ.setdefault("RAY_ADDRESS", "localhost:6379")
RAY_ADDRESS = os.environ["RAY_ADDRESS"]

import ray
import torch

from nnsight.modeling.vllm import VLLM


def wait_for_cluster(num_nodes=2, num_gpus=2, timeout=120, poll_interval=5):
    """Wait until the Ray cluster has the expected nodes and GPUs."""
    if not ray.is_initialized():
        # Join the remote cluster as a driver-only node so we get full
        # Ray runtime access (needed for vLLM compiled DAGs).
        try:
            ray.init(address="auto")
        except (ConnectionError, ValueError, RuntimeError):
            subprocess.run(
                ["ray", "start", f"--address={RAY_ADDRESS}",
                 "--num-gpus=0", "--num-cpus=0"],
                check=True, capture_output=True,
            )
            ray.init(address="auto")

    start = time.time()
    while time.time() - start < timeout:
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n["Alive"]]
        total_gpus = sum(n["Resources"].get("GPU", 0) for n in alive_nodes)

        print(
            f"Cluster: {len(alive_nodes)} alive nodes, {int(total_gpus)} GPUs "
            f"(waiting for {num_nodes} nodes, {num_gpus} GPUs)"
        )

        if len(alive_nodes) >= num_nodes and total_gpus >= num_gpus:
            print("Cluster ready!")
            for i, n in enumerate(alive_nodes):
                print(f"  Node {i}: {n['NodeManagerAddress']} GPUs={n['Resources'].get('GPU', 0)}")
            return

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Cluster did not reach {num_nodes} nodes / {num_gpus} GPUs within {timeout}s"
    )


def load_model():
    """Load GPT-2 with Ray executor, TP=2."""
    return VLLM(
        "gpt2",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.1,
        dispatch=True,
    )


def test_basic_logit(model):
    """Test basic logit access with cross-node TP."""
    prompt = "The Eiffel Tower is located in the city of"

    with model.trace(prompt, temperature=0.0, top_p=1):
        logits = model.logits.output.save()

    next_token = model.tokenizer.decode(logits.argmax(dim=-1))
    assert next_token == " Paris", f"Expected ' Paris', got '{next_token}'"


def test_intervention(model):
    """Test activation intervention across nodes."""
    prompt = "The Eiffel Tower is located in the city of"

    with model.trace(prompt, temperature=0.0, top_p=1):
        out = model.transformer.h[-2].mlp.output.clone()
        out[:] = 0
        model.transformer.h[-2].mlp.output = out
        hs = model.transformer.h[-2].mlp.output.save()
        logits = model.logits.output.save()

    assert torch.all(hs == 0), "Hidden states should be all zeros after intervention"


def test_multi_token_generation(model):
    """Test multi-token generation across nodes."""
    prompt = "Madison Square Garden is located in the city of"

    with model.trace(prompt, temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
        logits = list().save()
        with tracer.iter[0:3]:
            logits.append(model.logits.output)

    assert len(logits) == 3, f"Expected 3 logits, got {len(logits)}"


def test_generation_with_intervention(model):
    """Test intervention during multi-token generation across nodes."""
    prompt = "Madison Square Garden is located in the city of"

    with model.trace(prompt, temperature=0.0, top_p=1.0, max_tokens=5) as tracer:
        logits = list().save()
        hs_list = list().save()
        with tracer.iter[:] as it:
            if it == 2:
                out = model.transformer.h[-2].output.clone()
                out[0][:] = 0
                model.transformer.h[-2].output = out

            hs_list.append(model.transformer.h[-2].output[0])
            logits.append(model.logits.output)

    assert [torch.all(hs == 0) for hs in hs_list] == [
        False,
        False,
        True,
        False,
        False,
    ], "Only iteration 2 should have zeroed hidden states"


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Node NNsight vLLM Tests")
    print("=" * 60)

    print(f"\n[1/5] Waiting for Ray cluster at {RAY_ADDRESS}...")
    wait_for_cluster(num_nodes=2, num_gpus=2)

    print("\n[2/5] Loading model...")
    model = load_model()

    tests = [
        ("Basic logit access", test_basic_logit),
        ("Activation intervention", test_intervention),
        ("Multi-token generation", test_multi_token_generation),
        ("Generation with conditional intervention", test_generation_with_intervention),
    ]

    passed = 0
    failed = 0

    for i, (name, test_fn) in enumerate(tests, start=1):
        print(f"\n[Test {i}/{len(tests)}] {name}...")
        try:
            test_fn(model)
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")

    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    print("=" * 60)
    # Force-exit to avoid Ray shutdown hang
    os._exit(0 if failed == 0 else 1)
