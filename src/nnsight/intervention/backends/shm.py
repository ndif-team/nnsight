"""Handing a payload between two processes on one machine without sending it anywhere.

An NDIF deployed with ``--singleton`` runs on the same host as its client, so the
payload does not have to travel: ``/dev/shm`` is a tmpfs, a file there is RAM
with a path, and both processes can address the same pages. The client writes a
segment and sends the *path* where it would otherwise have sent a multipart body,
and reads the result back the same way.

**Both ends import this module** — nnsight's singleton backend and NDIF's
singleton server. It lives here rather than in NDIF because nnsight already owns
the wire format (`RequestModel`, `ResponseModel`, `Status`), and a transport
whose two halves are defined in different repositories is a transport whose two
halves drift.

Nothing reaches it unless a caller asks for it by name (``remote="singleton"``),
so a client pointed at the public NDIF never touches any of this.

**Nothing here expires.** tmpfs has no TTL, and POSIX shared memory has no
refcount that frees on last close: a segment lives until something unlinks it or
the machine reboots, and because it is RAM a leaked one is memory you do not get
back. Every segment has exactly one owner responsible for unlinking it — the
*reader* — which is why [`read`][nnsight.intervention.backends.shm.read] unlinks
by default, and [`sweep`][nnsight.intervention.backends.shm.sweep] exists for the
case where that owner died before it could.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger("nnsight.shm")

#: Where segments live. Matched to the server's default; both ends have to agree,
#: and they only ever agree by both being on this machine.
ROOT = os.environ.get("NDIF_SHM_DIR", "/dev/shm")

#: Every file this module creates starts with it, and
#: [`sweep`][nnsight.intervention.backends.shm.sweep] will not touch anything that
#: lacks it. ``/dev/shm`` belongs to the whole machine — Ray's own plasma store
#: lives there — so a sweep by age alone would eventually delete someone else's
#: memory.
PREFIX = "ndif-"

#: How old an orphan has to be before a starting server reclaims it. Generous on
#: purpose: the cost of waiting is some tmpfs held a while longer, and the cost of
#: being wrong is deleting a live request's payload out from under it.
STALE_SECONDS = 3600.0


def available() -> bool:
    """Whether this machine has a shared-memory directory we can write."""
    return os.path.isdir(ROOT) and os.access(ROOT, os.W_OK)


def path_for(request_id: str, kind: str) -> str:
    return os.path.join(ROOT, f"{PREFIX}{kind}-{request_id}")


def write(data: bytes, request_id: str, kind: str = "request") -> str:
    """Put ``data`` in shared memory and return its path.

    Written under a temporary name and renamed into place, so the server cannot
    observe a half-written segment — which would not look like a race, it would
    look like a corrupt payload.
    """
    final = path_for(request_id, kind)
    staging = f"{final}.partial"

    with open(staging, "wb") as handle:
        handle.write(data)
        handle.flush()

    os.rename(staging, final)
    return final


def read(path: str, *, unlink: bool = True) -> bytes:
    """Read a segment, and by default unlink it.

    The reader owns the segment: the writer cannot know when the other side is
    done, so if the reader doesn't unlink, nobody does. Unlinking while the
    descriptor is open keeps the data readable and reclaims the memory as soon
    as this returns.
    """
    with open(path, "rb") as handle:
        if unlink:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        return handle.read()


def discard(path: Optional[str]) -> None:
    """Unlink a segment, tolerating one that is already gone.

    For the error paths: a request that failed server-side leaves a payload
    nobody is coming to read.
    """
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def sweep(max_age_seconds: float = STALE_SECONDS) -> List[str]:
    """Unlink this module's segments older than ``max_age_seconds``; return them.

    Run when a singleton server starts, because that is the moment an orphan is
    provably one: nothing it owns can predate the process that owns it. A client
    killed between writing its payload and the server reading it leaves one
    behind, and there is no other point where anybody would notice.

    Restricted to ``PREFIX``: ``/dev/shm`` is not ours.
    """
    reclaimed: List[str] = []
    cutoff = time.time() - max_age_seconds

    try:
        names = os.listdir(ROOT)
    except OSError:
        logger.warning(f"Could not scan {ROOT} for stale segments", exc_info=True)
        return reclaimed

    for name in names:
        if not name.startswith(PREFIX):
            continue
        path = os.path.join(ROOT, name)
        try:
            if os.stat(path).st_mtime >= cutoff:
                continue
            os.unlink(path)
        except FileNotFoundError:
            continue
        except OSError:
            logger.debug(f"Could not reclaim {path}", exc_info=True)
            continue
        reclaimed.append(path)

    if reclaimed:
        logger.info(
            f"Reclaimed {len(reclaimed)} stale shared-memory segment(s) from {ROOT}"
        )
    return reclaimed
