# Multi-Node NNsight vLLM with Ray

Run NNsight interventions on vLLM models distributed across multiple nodes using Ray tensor parallelism.

## How It Works

When you pass `distributed_executor_backend="ray"` to `VLLM(...)`, NNsight replaces vLLM's default Ray executor with `NNsightRayExecutor`. This custom executor handles three scenarios automatically:

### 1. Remote Ray Cluster (`RAY_ADDRESS` is set)

```
RAY_ADDRESS=head-node:6379 python my_script.py
```

The driver process joins the remote Ray cluster as a driver-only node (0 GPUs, 0 CPUs) and submits all GPU work to the cluster's existing nodes. The driver machine only needs network access to the Ray GCS port (6379) — it does **not** need to be on the head node or any cluster node.

Under the hood:
- `ray start --address=$RAY_ADDRESS --num-gpus=0 --num-cpus=0` joins the cluster
- `ray.init(address="auto")` connects via the local session
- vLLM placement groups and compiled DAGs run on the cluster's GPU nodes

### 2. Existing Local Ray Cluster (no `RAY_ADDRESS`)

```
ray start --head
python my_script.py
```

If a local Ray cluster is already running, the executor connects to it via `ray.init(address="auto")` and uses its GPUs.

### 3. No Cluster (no `RAY_ADDRESS`, no local Ray)

```
python my_script.py
```

The executor starts a fresh local Ray cluster via `ray.init()` with all local GPUs available. This is the default single-machine behavior.

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
    logits = model.logits.output.save()
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
