"""Top-level NDIF helpers.

- `register` — ship a local module with a remote request (serialize by value).
- `status` / [`is_model_running`][nnsight.ndif.is_model_running] — query the NDIF service and its models.
- [`compare`][nnsight.ndif.compare] / [`get_remote_env`][nnsight.ndif.get_remote_env] — diff the local Python environment
  against a server's; each server's environment is cached per host.

Tables render as plain text with optional ANSI color (matching
[`nnsight.intervention.backends.display`][nnsight.intervention.backends.display]); no ``rich`` dependency. Network
libraries are imported lazily so importing nnsight stays light.
"""

from __future__ import annotations

import os
import pkgutil
import sys
from enum import Enum
from importlib.metadata import distributions, packages_distributions
from typing import Any, Optional, Union

from .schema.config import CONFIG

# Packages whose local/remote mismatch is highlighted (interventions break subtly
# when these drift between client and server).
CRITICAL_PACKAGES = {"nnsight", "transformers", "torch"}


# --- module registration for remote execution ------------------------------


def register(module: Any) -> None:
    """Register a local module for serialization by value in remote execution.

    Code submitted to NDIF that imports a module not installed on the server
    would raise ``ModuleNotFoundError``. Registering the module ships its class
    and function *source* with the request so it's rebuilt server-side. Thin
    wrapper over ``cloudpickle.register_pickle_by_value`` (the serializer nnsight
    uses honors it).

    Args:
        module: The module object, or its name as a string.

    Example::

        import mymodule
        import nnsight
        nnsight.register(mymodule)           # or nnsight.register("mymodule")

        with model.trace("Hello", remote=True):
            out = mymodule.myfn(model).save()
    """
    from cloudpickle.cloudpickle import (
        _PICKLE_BY_VALUE_MODULES,
        register_pickle_by_value,
    )

    if isinstance(module, str):
        _PICKLE_BY_VALUE_MODULES.add(module)
    else:
        register_pickle_by_value(module)


# --- rendering helpers ------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_COLORS = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m", "dim": "\033[2m"}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _paint(text: str, color: Optional[str], *, bold: bool = False) -> str:
    if color is None or not _supports_color():
        return text
    prefix = _COLORS.get(color, "") + (_BOLD if bold else "")
    return f"{prefix}{text}{_RESET}" if prefix else text


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    row_colors: Optional[list[Optional[str]]] = None,
) -> str:
    """A minimal fixed-width table. Widths are measured on plain text, so ANSI
    color (applied after padding) doesn't disturb alignment."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells: list[str], color: Optional[str] = None, bold: bool = False) -> str:
        return "  ".join(
            _paint(str(cell).ljust(widths[i]), color, bold=bold)
            for i, cell in enumerate(cells)
        )

    out = [line(headers, bold=True), "  ".join("-" * w for w in widths)]
    for i, row in enumerate(rows):
        out.append(line(row, row_colors[i] if row_colors else None))
    return "\n".join(out)


# --- service / model status ------------------------------------------------

# NDIF deployment levels that count as "deployed" (COLD = downloaded, not up).
_DEPLOYED_LEVELS = {"HOT", "WARM"}
# application_state -> table color.
_STATE_COLOR = {"RUNNING": "green", "DEPLOYING": "yellow", "UNHEALTHY": "red"}


def resolve_host(host: Optional[str] = None) -> str:
    """The base URL of an NDIF server, in the one spelling nnsight uses for it.

    ``None`` means the configured host (``CONFIG.API.HOST``, read at call time so
    a later change takes effect). A trailing slash is dropped: paths are appended
    as ``f"{host}/env"``, and the per-host environment cache keys on this string,
    so ``"http://h:5001"`` and ``"http://h:5001/"`` must be one server.
    [`RemoteBackend`][nnsight.intervention.backends.remote.RemoteBackend] resolves
    its per-call host (``remote="http://host:port"``) through here too.
    """
    host = (host or CONFIG.API.HOST).strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid host URL: {host!r}; must start with http:// or https://"
        )
    return host


def _get(
    path: str,
    timeout: tuple[float, float] = (5.0, 30.0),
    headers: Optional[dict] = None,
    host: Optional[str] = None,
) -> dict:
    import httpx  # lazy: only for actual NDIF calls

    connect, read = timeout
    response = httpx.get(
        f"{resolve_host(host)}{path}",
        timeout=httpx.Timeout(connect, read=read),
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


class NdifStatus:
    """A view of NDIF's deployed models, with a formatted table on ``print``.

    ``deployments`` maps repo id -> ``{model_class, repo_id, revision, level,
    state}`` for each HOT/WARM model; ``status`` is the derived service state.
    Indexing/iteration delegate to ``deployments`` for convenience.
    """

    class Status(str, Enum):
        UP = "UP"
        REDEPLOYING = "REDEPLOYING"
        DOWN = "DOWN"

    _MESSAGE = {
        Status.UP: "NDIF Service: Up 🟢",
        Status.REDEPLOYING: "NDIF Service: Redeploying 🟡",
        Status.DOWN: (
            "NDIF Service: Down 🔴\nVisit our community support at "
            "https://discuss.ndif.us/ or try again later."
        ),
    }

    def __init__(self, deployments: dict) -> None:
        self.deployments = deployments
        self.status = self._derive_status()

    def _derive_status(self) -> "NdifStatus.Status":
        states = {d["state"] for d in self.deployments.values()}
        if "RUNNING" in states:
            return NdifStatus.Status.UP
        if "DEPLOYING" in states:
            return NdifStatus.Status.REDEPLOYING
        return NdifStatus.Status.DOWN

    def __getitem__(self, key: str) -> dict:
        return self.deployments[key]

    def __iter__(self):
        return iter(self.deployments)

    def __contains__(self, key: str) -> bool:
        return key in self.deployments

    def __len__(self) -> int:
        return len(self.deployments)

    def keys(self):
        return self.deployments.keys()

    def __str__(self) -> str:
        rows, colors = [], []
        for value in self.deployments.values():
            rows.append([
                value["model_class"],
                value["repo_id"],
                value["revision"],
                value["level"],
                value["state"],
            ])
            colors.append(_STATE_COLOR.get(value["state"]))
        table = _render_table(
            ["Model Class", "Repo ID", "Revision", "Level", "State"], rows, colors
        )
        return f"{self._MESSAGE[self.status]}\n\n{table}"

    def __repr__(self) -> str:
        return self.__str__()


def status(raw: bool = False) -> Union[dict, NdifStatus]:
    """Query the NDIF service and its deployed models.

    Printing the returned [`NdifStatus`][nnsight.ndif.NdifStatus] shows a table of the currently
    deployed (HOT/WARM) models and their state.

    Args:
        raw: If ``True``, return the raw ``/status`` JSON instead of an
            [`NdifStatus`][nnsight.ndif.NdifStatus].

    Returns:
        The raw dict (``raw=True``), else an [`NdifStatus`][nnsight.ndif.NdifStatus] (empty, with a
        ``DOWN`` status, if the service is unreachable).

    Examples:
        >>> import nnsight
        >>> print(nnsight.status())          # table of deployed models
        >>> "openai-community/gpt2" in nnsight.status()
    """
    try:
        response = _get("/status")
    except Exception:
        print(NdifStatus._MESSAGE[NdifStatus.Status.DOWN], file=sys.stderr)
        return {} if raw else NdifStatus({})

    if raw:
        return response

    deployments = {}
    for name, value in response.get("deployments", {}).items():
        if value.get("deployment_level") not in _DEPLOYED_LEVELS:
            continue  # deployed models only (skip COLD downloaded)
        model_key = value.get("model_key", "")
        model_class = model_key.split(":", 1)[0].split(".")[-1] if model_key else "-"
        repo_id = value.get("repo_id", name)
        deployments[repo_id] = {
            "model_class": model_class,
            "repo_id": repo_id,
            "revision": value.get("revision") or "main",
            "level": value.get("deployment_level"),
            "state": value.get("application_state", "UNHEALTHY"),
        }
    return NdifStatus(deployments)


def ndif_status(raw: bool = False) -> Union[dict, NdifStatus]:
    """Deprecated: use `status`."""
    import warnings

    from . import NNsightDeprecationWarning

    warnings.warn(
        "nnsight.ndif_status() is deprecated; use nnsight.status() instead.",
        NNsightDeprecationWarning,
        stacklevel=2,
    )
    return status(raw)


def is_model_running(repo_id: str, revision: str = "main") -> bool:
    """Whether ``repo_id`` (at ``revision``) is currently RUNNING on NDIF.

    Returns ``False`` if the service is unreachable. The repo id is canonicalized
    via the Hub so different spellings match the deployed key.

    Examples:
        >>> import nnsight
        >>> nnsight.is_model_running("openai-community/gpt2")
    """
    try:
        response = _get("/status")
    except Exception:
        print(NdifStatus._MESSAGE[NdifStatus.Status.DOWN], file=sys.stderr)
        return False

    from huggingface_hub import HfApi  # lazy

    repo_id = HfApi().model_info(repo_id).id
    for value in response.get("deployments", {}).values():
        if value.get("repo_id") == repo_id and (value.get("revision") or "main") == revision:
            return value.get("application_state") == "RUNNING"
    return False


# --- authentication --------------------------------------------------------


def whoami(api_key: Optional[str] = None) -> dict:
    """Resolve the identity NDIF associates with an API key.

    Calls the service's ``/whoami`` endpoint and returns ``{"email": ..., "tags":
    [...]}``. ``email`` is ``None`` when the key is unrecognized (or the server has
    key validation disabled). Uses the configured key (``CONFIG.API.APIKEY``) when
    ``api_key`` isn't given.

    Args:
        api_key: The key to resolve; defaults to the configured one.

    Returns:
        The ``/whoami`` identity dict, ``{"email": str | None, "tags": list}``.
    """
    key = api_key or CONFIG.API.APIKEY
    return _get("/whoami", headers={"ndif-api-key": key} if key else None)


def login(api_key: Optional[str] = None) -> None:
    """Store your NDIF API key so future sessions can use it (HuggingFace-style).

    Prompts for the key with ``getpass`` (never echoed) when not given, verifies it
    against the service via [`whoami`][nnsight.ndif.whoami], then persists it with
    ``CONFIG.set_default_api_key`` (which writes ``config.yaml``). Verification is
    best-effort — the key is still saved if the service can't be reached or doesn't
    recognize it, with a note — so a typo is surfaced without blocking login.

    Args:
        api_key: The NDIF API key. Prompted for (hidden) when not given; empty input
            is a no-op that saves nothing.

    Examples:
        >>> from nnsight import login
        >>> login()
        Enter your NDIF API key:
        NDIF API key saved — logged in as you@example.com.
    """
    from getpass import getpass

    if api_key is None:
        api_key = getpass("Enter your NDIF API key: ")
    api_key = (api_key or "").strip()
    if not api_key:
        print("No API key provided. Nothing was saved.")
        return

    email = None
    try:
        email = whoami(api_key).get("email")
    except Exception as error:  # unreachable host, 404 on an older service, 503, ...
        print(f"Could not verify the key against {CONFIG.API.HOST} ({error}).")

    CONFIG.set_default_api_key(api_key)
    if email:
        print(f"NDIF API key saved — logged in as {email}.")
    else:
        print("NDIF API key saved (unverified).")


def main() -> None:
    """Console entry point for the ``nnsight`` command (``login`` / ``whoami``)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="nnsight", description="Command line tools for nnsight."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("login", help="Store your NDIF API key.")
    subparsers.add_parser("whoami", help="Show the identity for the stored API key.")

    args = parser.parse_args()

    if args.command == "login":
        login()
    elif args.command == "whoami":
        email = whoami().get("email")
        print(email if email else "Not logged in (no recognized API key).")
    else:
        parser.print_help()


# --- local vs remote environment comparison --------------------------------

# Server environments by resolved host URL (see `resolve_host`). One process can
# talk to several servers — the configured one and any per-call
# ``remote="http://host:port"`` — and each has its own packages, so the cache is
# per host. Seed or empty it with `set_remote_env` / `clear_remote_env`.
_REMOTE_ENVS: dict[str, dict] = {}


def get_local_env() -> dict:
    """The local Python version and installed packages (by import name)."""
    # dist name -> import names, so versions key by how you'd `import` them.
    dist_to_imports: dict[str, list[str]] = {}
    for import_name, dist_names in packages_distributions().items():
        for dist_name in dist_names:
            dist_to_imports.setdefault(dist_name, []).append(import_name)

    packages: dict[str, str] = {}
    for dist in distributions():
        dist_name = dist.metadata["Name"]
        import_names = dist_to_imports.get(dist_name, [])
        if import_names:
            for name in import_names:
                packages[name] = dist.version
        else:
            packages[dist_name] = dist.version

    # Local (non-site-packages) modules importable from sys.path, marked "local".
    site_paths = {p for p in sys.path if "site-packages" in p or "dist-packages" in p}
    for importer, module_name, _ in pkgutil.iter_modules():
        if module_name in packages or module_name.startswith("_"):
            continue
        if module_name in sys.stdlib_module_names:
            continue
        path = getattr(importer, "path", None)
        if path is not None and not any(path.startswith(sp) for sp in site_paths):
            packages[module_name] = "local"

    return {"python_version": sys.version, "packages": packages}


def get_remote_env(host: Optional[str] = None, *, force_refresh: bool = False) -> dict:
    """An NDIF server's Python version and installed packages, cached per host.

    Returns ``{"python_version": str, "packages": {import_name: version}}`` as the
    server's ``/env`` reports it. Each host is fetched once per process; a fetch
    that fails caches nothing, so the next call tries again.

    Args:
        host: The server's base URL — the same string a per-call
            ``remote="http://host:port"`` takes. Defaults to the configured host
            (``CONFIG.API.HOST``).
        force_refresh: Fetch again even if this host's environment is cached.

    Raises:
        RuntimeError: The host is unreachable or does not serve ``/env``; the
            message names the host.

    Examples:
        >>> from nnsight import ndif
        >>> ndif.get_remote_env()["packages"]["torch"]
        >>> ndif.get_remote_env("http://localhost:5001")["packages"].get("nnterp")
    """
    # `get_remote_env(True)` is the positional spelling of force_refresh; a bool is
    # never a host, so read it as the flag rather than failing on it as a URL.
    if isinstance(host, bool):
        host, force_refresh = None, host
    host = resolve_host(host)
    if force_refresh or host not in _REMOTE_ENVS:
        _REMOTE_ENVS[host] = _fetch_remote_env(host)
    return _REMOTE_ENVS[host]


def _fetch_remote_env(host: str) -> dict:
    import httpx  # lazy: only for actual NDIF calls

    try:
        return _get("/env", timeout=(5, 60), host=host)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            raise RuntimeError(
                f"The NDIF server at {host} does not serve /env, so its Python "
                "environment cannot be read."
            ) from error
        raise RuntimeError(
            f"Could not read the environment of the NDIF server at {host}: {error}"
        ) from error
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Could not reach the NDIF server at {host} to read its environment: {error}"
        ) from error


def set_remote_env(env: dict, host: Optional[str] = None) -> None:
    """Seed the cached environment for ``host`` (default: the configured host).

    [`get_remote_env`][nnsight.ndif.get_remote_env] and [`compare`][nnsight.ndif.compare] then answer for that host
    without a network call — the seam for tests and offline runs. ``env`` has the
    ``/env`` shape: ``{"python_version": str, "packages": {import_name: version}}``.
    """
    _REMOTE_ENVS[resolve_host(host)] = env


def clear_remote_env(host: Optional[str] = None) -> None:
    """Forget the cached environment for ``host``, or for every host when ``None``."""
    if host is None:
        _REMOTE_ENVS.clear()
    else:
        _REMOTE_ENVS.pop(resolve_host(host), None)


_PULLED_ENV = False


def pull_env() -> None:
    """Auto-register local (non-installed) modules for serialize-by-value, once.

    The remote backend calls this before its first request: every module
    [`get_local_env`][nnsight.ndif.get_local_env] discovers as ``"local"`` (importable from the working
    tree, not a pip install) is passed to `register`, so its source ships
    with remote requests. Without it, remote code that imports a local module
    raises ``ModuleNotFoundError`` server-side. Cached via ``_PULLED_ENV`` so the
    (cheap but non-trivial) local-env scan runs only once per process.
    """
    global _PULLED_ENV
    if _PULLED_ENV:
        return
    for package, version in get_local_env().get("packages", {}).items():
        if version == "local":
            register(package)
    _PULLED_ENV = True


def build_table(local_env: dict, remote_env: dict) -> str:
    """A package version-comparison table (local vs remote) as plain text."""
    local = local_env.get("packages", {})
    remote = remote_env.get("packages", {})

    # Critical packages first, then alphabetical; only packages the server has.
    def sort_key(pkg: str) -> tuple[bool, str]:
        return (pkg.lower() not in CRITICAL_PACKAGES, pkg.lower())

    rows, colors = [], []
    for pkg in sorted(remote, key=sort_key):
        local_ver = local.get(pkg, "-")
        remote_ver = remote[pkg]
        if local_ver == remote_ver:
            color, mark = "dim", "✓"
        elif pkg.lower() in CRITICAL_PACKAGES:
            color, mark = "red", "⚠ CRITICAL"
        else:
            color, mark = "yellow", "≠"
        rows.append([pkg, local_ver, remote_ver, mark])
        colors.append(color)

    return _render_table(
        ["Package", "Local Version", "Remote Version", "Status"], rows, colors
    )


class EnvComparison:
    """A local-vs-remote environment diff, inspectable and printable.

    Printing shows the Python-version line and the package table. Inspect the
    structured result directly:

    - ``local`` / ``remote`` — the raw env dicts.
    - ``local_python`` / ``remote_python`` / ``python_matches``.
    - ``packages`` — ``{pkg: {"local", "remote", "match", "critical"}}`` for every
      package the server has.
    - ``mismatches`` / ``critical_mismatches`` — the differing subset.
    """

    def __init__(self, local_env: dict, remote_env: dict) -> None:
        self.local = local_env
        self.remote = remote_env
        self.local_python = local_env.get("python_version", "Unknown").split()[0]
        self.remote_python = remote_env.get("python_version", "Unknown").split()[0]

        local_pkgs = local_env.get("packages", {})
        remote_pkgs = remote_env.get("packages", {})
        # Keyed by the packages the server actually has (a local-only package
        # can't cause a remote mismatch).
        self.packages = {
            pkg: {
                "local": local_pkgs.get(pkg, "-"),
                "remote": remote_pkgs[pkg],
                "match": local_pkgs.get(pkg, "-") == remote_pkgs[pkg],
                "critical": pkg.lower() in CRITICAL_PACKAGES,
            }
            for pkg in remote_pkgs
        }

    @property
    def python_matches(self) -> bool:
        return self.local_python == self.remote_python

    @property
    def mismatches(self) -> dict:
        return {p: v for p, v in self.packages.items() if not v["match"]}

    @property
    def critical_mismatches(self) -> dict:
        return {p: v for p, v in self.mismatches.items() if v["critical"]}

    def __str__(self) -> str:
        py_color, py_mark = ("dim", "✓") if self.python_matches else ("yellow", "≠")
        lines = [
            _paint("Python Version:", None, bold=True),
            _paint(f"  Local:  {self.local_python}", py_color),
            _paint(f"  Remote: {self.remote_python} {py_mark}", py_color),
            "",
            build_table(self.local, self.remote),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


def compare(host: Optional[str] = None) -> EnvComparison:
    """Compare the local Python environment with an NDIF server's.

    Package or Python-version drift between client and server can make
    interventions behave differently remotely than locally; this surfaces it.

    Returns an [`EnvComparison`][nnsight.ndif.EnvComparison] — ``print`` it for the table, or inspect
    ``.mismatches`` / ``.critical_mismatches`` / ``.python_matches``.

    Args:
        host: The server to compare against, as [`get_remote_env`][nnsight.ndif.get_remote_env] takes it.
            Defaults to the configured host.

    Examples:
        >>> import nnsight
        >>> print(nnsight.compare())         # local vs remote package table
        >>> nnsight.compare().critical_mismatches
        >>> nnsight.compare("http://localhost:5001").packages["nnterp"]
    """
    return EnvComparison(get_local_env(), get_remote_env(host))
