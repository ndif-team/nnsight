# Multi-Node NNsight vLLM with Ray

Run NNsight interventions on vLLM models distributed across multiple nodes using Ray tensor parallelism.

## How It Works

When you pass `distributed_executor_backend="ray"` to `VLLM(...)`, nnsight uses
**vLLM's own (stock) Ray executor** and injects its intervention logic through
vLLM's supported `worker_cls` hook (`NNsightGPUWorker`) — the same mechanism the
single-process / multiprocessing path uses. nnsight does **not** replace or
reimplement vLLM's executor, placement, or rank assignment; vLLM owns all of
that, which keeps the worker↔rank↔KV-cache-config mapping consistent.

> Historical note: an earlier `NNsightRayExecutor` fork *did* replace vLLM's
> executor (to add remote-driver support and work around a vLLM 0.15.1 actor
> import crash). It reimplemented vLLM's placement/rank machinery and
> intermittently mis-assigned per-stage KV-cache configs under PP (a ~50% flaky
> `KeyError` at load). It was removed in favor of the stock executor; the
> import-crash workaround is unnecessary on vLLM 0.19.1.

Cluster connection is handled by vLLM/Ray natively:
- **No cluster running:** vLLM starts a local Ray cluster using all local GPUs (default single-machine behavior).
- **Existing local Ray cluster:** vLLM connects to it (`ray.init(address="auto")`).
- **Remote cluster:** set `RAY_ADDRESS=head-node:6379` (or run the driver on a cluster node) and vLLM places workers across the cluster's nodes. (A driver on a machine with no GPUs is not exercised by this example; run the driver on a cluster node, as the Docker setup below does.)

## Quick Start with Docker (Simulated Multi-Node)

This example uses Docker containers to simulate multiple Ray nodes on a single machine. Each container gets one GPU and acts as a separate node, forcing cross-node NCCL communication over TCP.

### Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- 2+ GPUs
- nnsight installed on the host (for running the test script)

### 1. Configure

Edit `docker-compose.yml`:

- Set `device_ids` for each service to free GPUs on your machine
- Optionally set `HF_CACHE` to your HuggingFace cache directory

```bash
# Optional: point to your HF cache so models aren't re-downloaded in containers
export HF_CACHE=~/.cache/huggingface
```

### 2. Start the Cluster

```bash
docker compose up -d
```

This starts:
- **head** (GPU 0): Ray head node
- **worker** (GPU 1): Ray worker node

Wait ~10 seconds for both nodes to register.

### 3. Run Tests

```bash
python test_multinode.py
```

The test script:
1. Joins the Docker Ray cluster as a driver-only node
2. Loads GPT-2 with `tensor_parallel_size=2` across the two containers
3. Runs 4 tests validating cross-node interventions

Expected output:
```
[Test 1/4] Basic logit access...        PASSED
[Test 2/4] Activation intervention...   PASSED
[Test 3/4] Multi-token generation...    PASSED
[Test 4/4] Generation with conditional intervention... PASSED
ALL TESTS PASSED
```

### 4. Tear Down

```bash
docker compose down
ray stop  # clean up local driver node
```

## Running Your Own Multi-Node Setup

### With a Real Ray Cluster

If you have a Ray cluster already running (e.g., via `ray start --head` on the head node and `ray start --address=head:6379` on workers):

```python
import os
os.environ["RAY_ADDRESS"] = "head-node:6379"

from nnsight.modeling.vllm import VLLM

model = VLLM(
    "meta-llama/Llama-3.1-8B",
    tensor_parallel_size=4,
    distributed_executor_backend="ray",
)

with model.trace("The capital of France is", temperature=0.0, max_tokens=10):
    logits = model.logits.save()
    hidden = model.model.layers[15].output[0].save()

print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

### Key Points

- `RAY_ADDRESS` must be a **GCS address** (`host:6379`), not a Ray Client address (`ray://host:10001`). vLLM's compiled DAGs require full Ray runtime access.
- The `tensor_parallel_size` must match the total number of GPUs you want to use across the cluster.
- nnsight must be installed on all Ray worker nodes.
- Set `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` on worker nodes if GPUs are in separate containers or machines without NVLink/shared memory.

### Docker Environment Variables

The compose file sets these NCCL variables on all containers:

| Variable | Value | Why |
|----------|-------|-----|
| `NCCL_P2P_DISABLE` | `1` | NVLink doesn't work across containers/nodes |
| `NCCL_SHM_DISABLE` | `1` | `/dev/shm` isn't shared across containers |
| `NCCL_SOCKET_IFNAME` | `eth0` | Use the Docker bridge network for NCCL |
| `NCCL_DEBUG` | `INFO` | Log NCCL transport selection for debugging |

## Troubleshooting

**`RAY_ADDRESS must be a GCS address, not a Ray Client address`**
You used `ray://host:10001`. Change to `host:6379` (the GCS port).

**`Can't find node_ip_address.json` (60s timeout then works)**
This is expected on first connection. The driver attempts `ray.init(address="auto")`, which takes ~60s to timeout when there's no local session, then falls back to `ray start` to join the cluster.

**`Every node should have a unique IP address`**
The driver's IP doesn't match any cluster node. This should be handled automatically by the `VLLM_HOST_IP` fix, but if you see it, set `VLLM_HOST_IP` to the head node's IP manually.

**`NCCL timeout` or `NCCL connection refused`**
Check that `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` are set on all workers. Verify that containers/nodes can reach each other on the Docker bridge or cluster network.
