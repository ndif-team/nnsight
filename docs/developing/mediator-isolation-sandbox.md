# Mediator Isolation Sandbox — Design

**Status:** Draft / design (prototype scope) · **Date:** 2026-06-05 · **Author:** zikai
**Related:** `ndif` security regression suite (`src/services/ray/tests/security/`), `NDIF.md` §6–7,
nnsight `src/nnsight/intervention/interleaver.py` (Mediator / event protocol).

## 1. Motivation

NDIF executes **arbitrary user-submitted Python** on shared GPU infrastructure. Today that code runs
**in-process** with the model forward pass inside a single Ray `ModelActor`, behind an in-process
Python whitelist (`Protector`, `src/services/ray/src/ray/nn/security/protected_environment.py`).

The security regression suite establishes that this boundary is **architecturally unfixable in-process**:

- **10/10 in-process sandbox escapes succeed** (`().__class__.__mro__…__subclasses__()`, `str.format`
  globals walk, `ctypes`/`CDLL`, `torch.utils.cpp_extension.load`, `CodeType`, frame walking, pickle
  `__reduce__`, end-to-end `os.system` RCE). The literature consensus (pysandbox post-mortem; Modal,
  Lambda, Snowpark practice) is that containment must move to **OS-level isolation**.
- **8/8 multi-tenant Mediator-capability leaks succeed** (mutating `batch_group`, walking
  `mediator.interleaver.batcher.current_value`, poking sibling mediators, `swap(None, …)` poisoning all
  tenants). All share one root cause: the `Mediator` object is handed to user code with **mutable,
  reachable references** to scope and to the shared `Interleaver`/`Batcher`.

This document designs a refactor that **truly isolates each Mediator** from the model-hosting process,
so user code can no longer reach the host, the GPU, the weights, or other tenants' activations.

## 2. Core insight: the event protocol is already the boundary

A `Mediator` (`interleaver.py:718`) already communicates with the forward pass **only** through six
typed messages over two one-slot queues (`interleaver.py:338`, `:949`):

| Event | From | Means |
|---|---|---|
| `VALUE` | worker | "give me the value at this provider path" (worker blocks) |
| `SWAP` | worker | "replace the value at this provider path with this tensor" |
| `SKIP` | worker | "bypass module X's real forward" |
| `BARRIER` | worker | "synchronize me with these other mediators" |
| `END` | worker | "I'm finished — drain me" |
| `EXCEPTION` | worker | "I crashed; here's the exception" |

This is already a clean **client/server RPC**. It is insecure only because client and server share one
Python interpreter, one address space, and one GPU context — so the worker can ignore the protocol and
walk Python objects directly.

> **Isolation = keep the six messages, move the worker to the far side of a real OS boundary, and change
> the transport from an in-process queue to a socket.** Every in-process escape and every
> Mediator-capability leak then disappears *structurally* — there is no longer a `mediator` object, a
> `batcher`, or a host module in the user's address space to reach.

## 3. Threat model (scoped)

### In scope
- **A. Host protection** — user code must not reach the host OS, filesystem, network, model weights, or GPU.
- **B. Cross-tenant protection** — when tenants are co-batched in one forward pass (lands with
  `ndif-batch`/`ndif-vllm`), tenant X must not read or poison tenant Y's rows. Solved two ways: (1) the
  worker holds *no host references*, and (2) the host-side `Batcher` becomes the **authority** on row
  bounds (below).
- **Deserialization RCE** — the pickle `__reduce__` gadget fires during `request.deserialize(...)` in
  `pre()` (`NDIF.md:838`), *before* execution, under a whitelist that necessarily includes
  `pickle`/`cloudpickle`. Untrusted bytes must therefore be unpickled **inside the jail**, never on the host.

### Out of scope (deliberately deferred)
- **Mediator pacing / forward-pass-blocking DoS** (suite leak #7). This is a *scheduler* property, not an
  isolation property. The host may keep blocking on the worker's reply exactly as today — just over a
  socket. Revisit with the non-blocking-listener work, separately.
- **Side channels** (timing, cache, GPU memory residency).
- **Per-hook transport throughput.** Accepted as a cost for the prototype; see §7 (option D) for the
  optimization path.

### Trust boundary
- **Trusted:** model weights, GPU + CUDA context, forward pass, `Interleaver`, `Batcher`, host OS, other
  tenants' data.
- **Untrusted:** the serialized request bytes (pickle), the compiled intervention function, anything the
  user can influence.

## 4. Design

### 4.1 Process / thread model

```
┌─ ModelActor (TRUSTED: GPU + weights) ───────────────────────┐
│  main thread: forward pass                                  │
│  Interleaver + Batcher   ← server-side row-bound authority  │
│  per request: MediatorProxy (host stub, speaks 6 events)    │
│  warm pool of CPU-only jailed workers (torch+nnsight ready) │
└───────────────┬─────────────────────────────────────────────┘
                │ per-request bidirectional socket (fd passed into jail)
┌───────────────┴─ jailed worker (UNTRUSTED, CPU-only) ───────┐
│  1. deserialize raw request bytes  ← untrusted pickle HERE  │
│  2. run compiled intervention fn → Mediator worker          │
│  3. VALUE/SWAP/SKIP/BARRIER/END/EXCEPTION over the socket   │
│  net=none · ro-fs allowlist · non-root · mem cap · no GPU   │
└──────────────────────────────────────────────────────────────┘
```

- The host keeps the forward pass, `Interleaver`, and `Batcher`. Per active request it spins a
  **`MediatorProxy`** — what is today the in-process worker thread, demoted to a transport endpoint. It
  receives the worker's events and answers them via `Batcher.narrow`/`swap`, **bounded to that tenant's
  admission-time rows.**
- **The protocol is unchanged.** Only the transport relocates: `Mediator.start` launches a jailed
  process instead of a `Thread`; the `Mediator.Value` one-slot queues (`interleaver.py:767`) become a
  socket. `Mediator.handle` writes a value to the socket instead of `response_queue`; the worker's
  `request(...)` reads/writes the socket instead of the in-process queues. `iter`, `barrier`, `next`,
  and save-filtering keep working with no user-visible change.

### 4.2 The activation data path (the accepted cost)

`VALUE`: host `narrow`s to the tenant's rows → `D2H` copy → serialize CPU tensor → socket → jail
deserializes → user gets a **real CPU tensor** and runs arbitrary torch-CPU on it. `SWAP`: jail sends
the modified CPU tensor back → host deserializes → `H2D` → writes into the **bounded** batch slice.

Start with the simplest serializer (raw buffer / `safetensors` over the socket); a shared-memory ring is
a later optimization, not part of correctness. This is **the per-hook D2H/H2D transport** — clean isolation, transport cost
accepted. See §7 for why and when to move to the two-tier approach.

### 4.3 Batcher as server-side authority (Boundary B)

Row bounds are recorded on the **host** at admission. The worker's `VALUE` carries only a provider path +
iteration — never a `batch_group` it can forge. `narrow(None)`, `[-1, _]`, and widening
(`batching.py:213–217`) become **unrepresentable** because the bounds live on the trusted side. This is
the one piece of genuinely new logic; everything else is transport relocation.

### 4.4 CPU-only, and why

The jail gets **no GPU**. This is load-bearing for both security and the warm pool:

- No CUDA context in the jail → no GPU-driver attack surface, and no CUDA-IPC cross-tenant leak (CUDA IPC
  shares at *allocation* granularity, which under vLLM's paged allocator can span other tenants' KV cache).
- **torch-CPU is fork-safe**, so a *zygote* that has already imported torch+nnsight can fork a fresh
  child per request in ~ms. (A CUDA-initialized process is **not** fork-safe — GPU-in-jail would kill the
  pool and is rejected.)

### 4.5 Lifecycle, pool, cancellation

- **One jail per request**, assigned to exactly one tenant, then **destroyed / re-exec'd** — never
  soft-reused across tenants (a dirtied interpreter is a cross-tenant channel).
- **Warm pool:** first cut may spawn-per-request; then add a zygote pool (pre-imported, pre-jailed idle
  workers claimed-then-destroyed) for sub-100ms starts.
- **Cancellation / timeout** becomes **`SIGKILL` the subprocess** instead of the current
  `kill_thread()` `ctypes`-injected `SystemExit` (`NDIF.md:919`) — a free robustness win.

### 4.6 Minimal surgery footprint

Three touch points, which is what makes this prototype-friendly:
1. `Mediator.start` + the `Value` queue handoff → jailed process + socket.
2. `pre()` deserialization moves inside the jail.
3. `Batcher.narrow`/`swap` take host-held bounds instead of trusting the message.

The forward pass, interleaver scheduling, and nnsight's public API are untouched.

## 5. Isolation mechanism

Three tiers, distinguished by **what enforces the boundary**:

| Tier | Mechanism | Tools | Host-kernel surface |
|---|---|---|---|
| 1 | Linux namespaces + seccomp + cgroups (shared kernel) | bubblewrap, nsjail, runc/Docker | full (filtered) |
| 2 | User-space kernel (syscalls reimplemented) | gVisor (`runsc`) | minimal |
| 3 | microVM (own guest kernel, hardware virt) | Firecracker | none |

**Choice for the prototype: Tier 1, CPU-only.** Compute-bound torch-CPU runs ~native; the socket
transport is light; one jail per request is cheap. gVisor (Tier 2) is the **pre-planned upgrade** behind
the same launcher interface if "shared host kernel" becomes unacceptable. Firecracker is overkill for a
CPU-only prototype.

**Tool: nsjail strategically; bubblewrap as the bootstrap.** The probe (§6) shows:
- `nsjail` is **not packaged** on either the host (Ubuntu 20.04) or the container (Debian trixie) — it is
  a source build. Its value is integrated seccomp-policy (kafel) + cgroups + rlimits + time limits, which
  is what a production sandbox wants.
- `bubblewrap` (0.11.0) is a **one-line apt install** in the container and is **verified working** on the
  host. It is a launcher (pair it with an external seccomp filter + cgroup), but gives the *identical*
  Tier-1 boundary.
- The nesting constraints in §6 apply **identically to both**, so the bwrap-vs-nsjail choice is
  independent of — and less important than — the deployment decision in §6.5.

**Recommendation:** bootstrap on bwrap (verified, trivial install, swappable behind a `Jailer` interface),
target nsjail for the hardened path.

## 6. Host-reality probe (verified 2026-06-05)

All results are measured on this machine, not assumed.

### 6.1 Bare host (Ubuntu 20.04, kernel 5.15.0-139)

| Check | Result |
|---|---|
| `kernel.unprivileged_userns_clone` | `1` (enabled) |
| `user.max_user_namespaces` | `6185255` |
| `unshare -Urn true` (unpriv userns) | **OK** |
| `bwrap` full jail (userns+net+pid+fresh-proc+ro-fs) | **OK**, pid1=bwrap |
| Network isolation inside jail | **OK** (no DNS, loopback only) |
| ro-fs allowlist (unbound `/disk` invisible) | **OK** |
| `bwrap` present / `nsjail` present | `/usr/bin/bwrap` / **absent** |
| seccomp kernel support | `CONFIG_SECCOMP=y`, filter=y |
| cgroup version | v1 |

→ **Tier-1 unprivileged jailing is fully viable on the bare host.**

### 6.2 Inside `dev-ray-1` (local stand-in for the prod model container; Debian trixie, ndif/ndif:latest)

| Check | Result |
|---|---|
| uid inside container | `0` (root-in-container) |
| Privileged / CapAdd / SecurityOpt | `false` / `[]` / `[]` (default posture) |
| AppArmor | `docker-default` |
| `unshare -Urn` (nested unpriv userns) | **BLOCKED** — "Operation not permitted" |
| torch | `2.9.1+cu128`, cuda available |
| `bwrap` / `nsjail` present | **absent** / **absent** (bwrap = 1-line apt; nsjail = build) |

The block is **Docker's default seccomp profile** denying `clone(CLONE_NEWUSER)`; the container also
lacks `CAP_SYS_ADMIN`, so classic root-namespaces are unavailable too. Both jail paths are closed by default.

### 6.3 What it takes to jail *inside* the container

| Container config | Result |
|---|---|
| default | userns BLOCKED |
| `seccomp=unconfined` | userns OK; `bwrap` fails at "make / slave" (AppArmor) |
| `seccomp=unconfined` + `apparmor=unconfined` | propagation OK; **fresh `/proc` mount fails** in new PID ns |
| `+ cap-add SYS_ADMIN` | still fails fresh `/proc` mount |
| `--privileged` | **full jail (fresh proc) works** |
| `seccomp+apparmor=unconfined`, **bind `/proc`, drop PID-unshare** | **jail builds** (reached execvp) — but **no PID-namespace isolation** |

**Finding:** nesting a userns mount-namespace jail inside an *unprivileged* container is fragile. Locked
container mounts and the fresh-`/proc` restriction mean a jail with **real PID isolation requires
`--privileged`** (or close), and the non-privileged workaround (bind the container's `/proc`) **leaks PID
visibility** — a cross-tenant info-leak vector if multiple workers share the container's PID space.

### 6.4 Production target: AWS ECS-on-EC2 (the probe transfers)

Production runs on **AWS ECS-on-EC2** (CDK `ndif-aws/service_stack.py:275`, `ecs.Ec2TaskDefinition`) —
not Fargate, not raw EC2. The model server (`Ray-Worker`) is an **ECS task (Docker container)** on a
**`g4dn.xlarge`** GPU instance (1× T4, **4 vCPU / 16 GB**; `env/dev.py`), on the **ECS-optimized Amazon
Linux 2023 GPU AMI** (`service_stack.py:220`). The whole GPU is allocated into the container
(`gpu_count`, `:365`); network mode is `awsvpc` (own ENI per task); models live on EFS (`/efs/huggingface`).

Decisive detail: the task's `linux_parameters` is used **only** for `shared_memory_size`
(`service_stack.py:351-355`) — **no `privileged`, no added caps, no seccomp/apparmor override.** So the
prod model container runs the **Docker default posture, identical to local `dev-ray-1`.**

→ **The §6.1–6.3 probe transfers to production unchanged:** nested unprivileged userns is blocked in prod
too (docker-default seccomp + no `CAP_SYS_ADMIN`, GPU inside the container). The local finding was
representative, not a local artifact.

Two AWS specifics change the *decision* (§6.5):
- **ECS has no clean per-task seccomp knob.** ECS `dockerSecurityOptions` covers SELinux/AppArmor labels
  only — not seccomp profiles. So the local "`seccomp=unconfined`" relaxation is **not** a per-task option
  on ECS; the practical in-task lever is **`privileged=true`** (drops seccomp *and* adds all caps).
  The in-container self-jail is therefore *heavier* on AWS than locally.
- **We own the EC2 instance, its user-data, and the host Docker daemon.** That makes the sibling approach
  host-level-sibling deployment natural: pre-install the jailer in user-data and run sandboxes as **host-level siblings**, leaving
  the model-server task unprivileged.

*Adjacent finding (out of scope for this design, worth fixing): `ndif-aws/env/dev.py` commits **live
credentials** — an AWS access key/secret, a HuggingFace token, DB/Influx passwords. Rotate and move to SSM
Parameter Store / Secrets Manager.*

## 6.5 Deployment decision (KEY OPEN ITEM)

The nesting reality reshapes *where* the jail runs. Two resolutions, read through the AWS reality (§6.4):

- **Self-jail inside a loosened model container.** Locally this is
  `seccomp=unconfined, apparmor=unconfined` (+ effectively `privileged` for real PID isolation).
  **On ECS it collapses to `privileged=true`** — there is no clean per-task seccomp knob — set via the
  container's `linux_parameters`/props (`service_stack.py:351-363`). Fastest to a testable boundary, **but
  on AWS it means a privileged model task**, which broadens that task's own surface more than the local
  case did. Justified only as a throwaway first cut to exercise the protocol.
- **Sibling sandbox on the host (recommended target, especially on AWS).** Run the jailed worker as a
  **separate process/container at the EC2-instance level**, where host userns + bwrap work (verified on the
  local host §6.1; AL2023 host expected to match — needs a one-line on-instance confirm). Pre-install the
  jailer in EC2 user-data; the model-server ECS task stays **unprivileged** and talks to the sandbox over a
  socket. This is how production code-exec services isolate (sibling sandboxes, not nested jails). Cost: a
  host-level launcher/broker and its trust handling (don't hand the host Docker socket to the model server).

**Recommendation:** **the host-level sibling jail is the production target** — owning the EC2 host makes it clean and avoids a
privileged model task. The in-container self-jail is justified only as a throwaway first cut; on ECS even that costs
`privileged`. Keep the launcher abstracted so in-container → host-level (and basic-seccomp → gVisor) don't touch anything above it.

**Sizing caveat:** `g4dn.xlarge` is **4 vCPU / 16 GB**, shared by Ray + the model + any CPU-only sandbox
pool. That caps pool size and the per-hook D2H/H2D budget (the per-hook D2H/H2D transport), pushing the two-tier optimization
(§7) *earlier* on small prod instances than a pure-correctness view suggests.

## 7. Future optimization (out of scope now)

Where do user tensor ops execute? The prototype picks **per-hook D2H/H2D in the jail, on CPU** (D2H/H2D per hook).
The destination is **the two-tier approach**: run the common tiny tensor algebra (read/write/project/ablate/steer/
patch/topk/cache) on the host via a *validated op interpreter* (data never leaves the GPU, no jail in the
hot path), and route only genuinely-arbitrary Python to the CPU jail. (B) GPU-in-jail via CUDA IPC is
rejected (surface + breaks the pool); (C) fully symbolic host execution is a breaking change to nnsight's
"real tensors" contract. The two-tier approach attacks the root — *arbitrary code next to the data* — rather than paying to
isolate code that didn't need to be arbitrary.

On small prod instances this is **less optional than it sounds**: a `g4dn.xlarge` (4 vCPU / 16 GB, §6.4)
leaves little headroom for a CPU-only sandbox pool doing per-hook D2H/H2D, so the two-tier approach likely graduates from
"later optimization" to "needed for throughput" before this ships at scale.

## 8. Decisions & open questions

**Decided**
- Move the worker across the existing six-event protocol boundary; change transport only.
- Tier-1 isolation, CPU-only, one jail per request, host-side `Batcher` as row-bound authority,
  deserialize-in-jail, `SIGKILL` cancellation.
- Prototype tensor path = per-hook D2H/H2D transport; destination = the two-tier approach — and on small prod instances (`g4dn`, §6.4) the two-tier approach is
  needed sooner than "later".
- Bootstrap on bubblewrap, target nsjail, gVisor as the strong-boundary upgrade.
- **Prod runs ECS-on-EC2** (§6.4); the probe transfers (same Docker-default posture). **the host-level sibling jail (host-level
  sibling on the EC2 instance) is the production target**; in-container the in-container self-jail is a throwaway first cut and on
  ECS costs `privileged=true`.

**Open**
- **in-container vs host-level sequencing** (§6.5) — the host-level sibling jail is the target; the only question is whether a privileged in-container first
  cut is worth building at all, or whether to go straight to the host-level sibling.
- Warm-pool mechanism: zygote-fork vs pre-launched idle-worker pool.
- Tensor wire format: raw buffer vs `safetensors` vs shmem ring (correctness first, optimize later).
- Exact `Jailer` interface so bwrap → nsjail → runsc are drop-in.
