from __future__ import annotations

import json
import pkgutil
import sys
import sysconfig
from enum import Enum
from importlib.metadata import distributions, packages_distributions
from io import StringIO
from sys import stderr
from typing import Union, types

import requests
from cloudpickle.cloudpickle import _PICKLE_BY_VALUE_MODULES, register_pickle_by_value
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table

from . import CONFIG


def register(module: types.ModuleType | str):
    """
    Register a local module for serialization by value when executing remotely on NDIF.

    When submitting code for remote execution on NDIF, any local modules that are not
    installed on the server will cause a ``ModuleNotFoundError``. This function registers
    a module so that its class definitions and function source code are serialized and
    sent along with the request, allowing them to be rebuilt on the server.

    This is a wrapper around ``cloudpickle.register_pickle_by_value``.

    Args:
        module: The module to register for serialization by value. Can be the actual
            module object or a string with the module's name.

    Note:
        - Call this function after importing the module but before using any of its
          contents in a remote context.
        - The module's source code and definitions will be included in the serialized
          payload, so keep registered modules reasonably sized.

        Examples:
        >>> import mymodule
        >>> from nnsight import LanguageModel
        >>> from nnsight.ndif import register
        >>>
        >>> # Register the local module so it can be used remotely
        >>> register(mymodule)
        >>>
        >>> # Or register by module name
        >>> register("mymodule")
        >>>
        >>> # Now you can use functions/classes from mymodule in remote execution
        >>> from mymodule.myfile import myfunction
        >>>
        >>> model = LanguageModel("meta-llama/Llama-3.1-70B")
        >>> with model.generate("Hello", remote=True):
        ...     result = myfunction(model)
        ...     result.save()
    """
    if isinstance(module, str):
        _PICKLE_BY_VALUE_MODULES.add(module)
    else:
        register_pickle_by_value(module)


# Critical packages that should be highlighted in red if mismatched
CRITICAL_PACKAGES = {"nnsight", "transformers", "torch"}


class NdifStatus(dict):
    """
    Status for remote execution on NDIF.

    This class provides a structured view of the NDIF status, including information
    about all deployed models and their current states. It inherits from dict, allowing
    direct access to the underlying status data while providing rich formatting for display.

    Attributes:
        status: The overall service status (UP, REDEPLOYING, or DOWN).

        Examples:
        >>> from nnsight import ndif_status
        >>> status = ndif_status()
        >>> print(status)  # Displays a formatted table of all models
        >>> status.status  # Returns NdifStatus.Status.UP, etc.
    """

    class Status(Enum):
        """
        Overall NDIF service status.

        Attributes:
            UP: Service is operational with at least one model running.
            REDEPLOYING: Service is transitioning; models are deploying or starting.
            DOWN: Service is unavailable or no models are running.
        """

        UP = "UP"
        REDEPLOYING = "REDEPLOYING"
        DOWN = "DOWN"

        @classmethod
        def _message(cls, status: NdifStatus.Status) -> str:
            if status == cls.UP:
                return "NDIF Service: Up 🟢"
            elif status == cls.REDEPLOYING:
                return "NDIF Service: Redeploying 🟡"
            elif status == cls.DOWN:
                return "NDIF Service: Down 🔴\nVisit our community support at https://discuss.ndif.us/ or try again later."

    class ModelStatus(Enum):
        """
        Status of an individual model deployment.

        Attributes:
            RUNNING: Model is fully deployed and accepting requests.
            DEPLOYING: Model is currently being deployed.
            NOT_DEPLOYED: Model is configured but not yet started.
            DOWN: Model deployment has failed or is unavailable.
        """

        RUNNING = "RUNNING"
        DEPLOYING = "DEPLOYING"
        NOT_DEPLOYED = "NOT DEPLOYED"
        DOWN = "DOWN"

        @classmethod
        def _color(cls, state: NdifStatus.ModelStatus) -> str:
            if state == cls.RUNNING:
                return "green"
            elif state == cls.DEPLOYING or state == cls.NOT_DEPLOYED:
                return "yellow"
            elif state == cls.DOWN:
                return "red"

    class DeploymentType(Enum):
        """
        Type of model deployment on NDIF.

        Attributes:
            DEDICATED: Model is a permanent deployment.
            PILOT_ONLY: Model available only for pilot users.
            SCHEDULED: Model runs on a schedule (e.g., specific hours).
        """

        DEDICATED = "Dedicated"
        PILOT_ONLY = "Pilot-Only"
        SCHEDULED = "Scheduled"

        @classmethod
        def _color(cls, deployment_type: NdifStatus.DeploymentType) -> str:
            if deployment_type == cls.DEDICATED:
                return "green"
            elif deployment_type == cls.PILOT_ONLY:
                return "purple"
            elif deployment_type == cls.SCHEDULED:
                return "blue"

    def __init__(self, response: dict):
        """
        Initialize NdifStatus with formatted response data.

        Args:
            response: Dictionary mapping repo_id to model info dictionaries,
                each containing 'model_class', 'repo_id', 'revision', 'type', and 'state'.
        """
        super().__init__(response)

        self._data = response
        self._table = self._table()
        self._status: NdifStatus.Status = self._status(response)

    @property
    def status(self) -> NdifStatus.Status:
        return self._status

    @classmethod
    def request_status(cls) -> Union[NdifStatus.Status, dict]:
        """
        Fetch raw status data from the NDIF API.

        Returns:
            The raw JSON response from the NDIF status endpoint.

        Raises:
            Exception: If the request times out, with a DOWN status message.
        """
        try:
            response = requests.get(f"{CONFIG.API.HOST}/status", timeout=(5, 30))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(NdifStatus.Status._message(NdifStatus.Status.DOWN))

        response = response.json()
        return response

    def _status(cls, response: dict) -> Status:
        if any(
            [
                value["state"] == NdifStatus.ModelStatus.RUNNING
                for value in response.values()
            ]
        ):
            return NdifStatus.Status.UP
        elif any(
            [
                value["state"] == NdifStatus.ModelStatus.DEPLOYING
                or value["state"] == NdifStatus.ModelStatus.NOT_DEPLOYED
                for value in response.values()
            ]
        ):
            return NdifStatus.Status.REDEPLOYING
        else:
            return NdifStatus.Status.DOWN

    def _table(self):
        table = Table()
        table.add_column("Model Class")
        table.add_column("Repo ID", no_wrap=True)
        table.add_column("Revision")
        table.add_column("Type")
        table.add_column("Status")

        for key, value in self.items():
            table.add_row(
                value["model_class"],
                value["repo_id"],
                value["revision"],
                f"[b {NdifStatus.DeploymentType._color(value['type'])}]{value['type'].value }[/]",
                f"[b {NdifStatus.ModelStatus._color(value['state'])}]{str(value['state'].value)}[/]",
            )

        return table

    def __str__(self):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, stderr=False)
        console.print(NdifStatus.Status._message(self.status), "\n", self._table)
        return buf.getvalue()

    def __repr__(self):
        return self.__str__()


def status(raw: bool = False) -> Union[dict, NdifStatus]:
    """
    Query the current status of the NDIF service and all deployed models.

    This is the primary function for checking NDIF availability and model status.
    When printed, the returned NdifStatus object displays a formatted table of
    all available models with their deployment type and current state.

    Args:
        raw: If True, return the raw API response dict. If False (default),
            return a formatted NdifStatus object.

    Returns:
        If raw=True: The raw JSON response from the NDIF API.
        If raw=False: An NdifStatus object with formatted model information,
            or an empty dict if the request fails.

        Examples:
        >>> from nnsight import ndif_status
        >>> status = ndif_status()
        >>> print(status)
        NDIF Service: Up 🟢
        ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ Model Class   ┃ Repo ID                    ┃ Revision ┃ Type      ┃ Status  ┃
        ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
        │ LanguageModel │ meta-llama/Llama-3.1-70B   │ main     │ Dedicated │ RUNNING │
        └───────────────┴────────────────────────────┴──────────┴───────────┴─────────┘
    """
    try:
        response = NdifStatus.request_status()
    except Exception as e:
        print(e, file=stderr)
        return {}

    if raw:
        return response
    else:
        formatted_response = {}
        for key, value in response["deployments"].items():
            if (
                value["deployment_level"] == "HOT"
                or value["deployment_level"] == "WARM"
                or "schedule" in value
            ):
                model_class = value["model_key"].split(":", 1)[0].split(".")[-1]
                hf_model = json.loads(value["model_key"].split(":", 1)[-1])
                repo_id = hf_model["repo_id"]
                revision = (
                    hf_model["revision"] if hf_model["revision"] is not None else "main"
                )

                state = NdifStatus.ModelStatus(
                    value.get("application_state", "NOT DEPLOYED")
                )
                if "dedicated" in value:
                    type = (
                        NdifStatus.DeploymentType.DEDICATED
                        if value["dedicated"] == True
                        else NdifStatus.DeploymentType.PILOT_ONLY
                    )
                elif "schedule" in value:
                    type = NdifStatus.DeploymentType.SCHEDULED

                formatted_response[repo_id] = {
                    "model_class": model_class,
                    "repo_id": repo_id,
                    "revision": revision,
                    "type": type,
                    "state": state,
                }

        return NdifStatus(formatted_response)

def ndif_status(raw: bool = False) -> Union[dict, NdifStatus]:
    """
    Deprecated: Use status() instead.
    """
    return status(raw)

def is_model_running(repo_id: str, revision: str = "main") -> bool:
    """
    Checks if a specific model is currently running on NDIF.

    Args:
        repo_id: The HuggingFace repository ID (e.g., "meta-llama/Llama-3.1-70B").
        revision: The model revision/branch to check. Defaults to "main".

    Returns:
        True if the model is running and available, False otherwise
        (including if the API request fails).

        Examples:
        >>> from nnsight import is_model_running
        >>> if is_model_running("meta-llama/Llama-3.1-70B"):
        ...     print("Model is available!")
    """
    try:
        response = NdifStatus.request_status()
    except Exception as e:
        print(e, file=stderr)
        return False

    repo_id = HfApi().model_info(repo_id).id

    for key, value in response["deployments"].items():
        if (
            value["repo_id"] == repo_id
            and (value["revision"] if value["revision"] is not None else "main")
            == revision
        ):
            return value.get("application_state", None) == "RUNNING"

    return False


NDIF_ENV = None


def get_local_env() -> dict:
    """Get the local Python environment information."""

    # Build reverse mapping: dist_name -> [import_names]
    pd_map = packages_distributions()
    dist_to_imports = {}
    for import_name, dist_names in pd_map.items():
        for dist_name in dist_names:
            if dist_name not in dist_to_imports:
                dist_to_imports[dist_name] = []
            dist_to_imports[dist_name].append(import_name)

    packages = {}
    for dist in distributions():
        dist_name = dist.metadata["Name"]
        version = dist.version

        # Get import names from packages_distributions mapping
        import_names = dist_to_imports.get(dist_name, [])

        if import_names:
            for imp_name in import_names:
                packages[imp_name] = version
        else:
            # Fallback to distribution name if no import mapping found
            packages[dist_name] = version

    # Get stdlib and site-packages paths to filter them out
    stdlib_path = sysconfig.get_paths()["stdlib"]
    site_packages_paths = set()
    for path in sys.path:
        if "site-packages" in path or "dist-packages" in path:
            site_packages_paths.add(path)

    # Discover local modules from sys.path (not in stdlib or site-packages)
    for importer, module_name, is_pkg in pkgutil.iter_modules():
        if module_name not in packages and not module_name.startswith("_"):
            # Check if this module comes from a local path
            if hasattr(importer, "path"):
                module_path = importer.path
                # Skip stdlib and site-packages
                if module_path.startswith(stdlib_path):
                    continue
                if any(module_path.startswith(sp) for sp in site_packages_paths):
                    continue
                # This is a local module
                packages[module_name] = "local"

    return {
        "python_version": sys.version,
        "packages": packages,
    }


def get_remote_env(force_refresh: bool = False) -> dict:
    """
    Fetch and cache the NDIF environment information from the remote server.

    Args:
        force_refresh: If True, always refetch even if cached.

    Returns:
        The environment info from NDIF.

    Raises:
        Exception: If the request fails.
    """
    global NDIF_ENV
    if not force_refresh and NDIF_ENV is not None:
        return NDIF_ENV
    try:
        response = requests.get(f"{CONFIG.API.HOST}/env", timeout=(5, 60))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch NDIF environment: {e}")
    NDIF_ENV = response.json()
    return NDIF_ENV


def _supports_color() -> bool:
    """
    Check if the terminal supports color output.

    Returns:
        True if color output is supported, False otherwise.
    """
    import os
    import sys

    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return True


_SUPPORTS_COLOR = _supports_color()


def build_table(local_env: dict, remote_env: dict) -> Table:
    """Build the environment comparison table for local and remote."""
    table = Table(
        title="NDIF Environment Comparison",
        show_lines=False,
        expand=True,
        highlight=_SUPPORTS_COLOR,
    )
    table.add_column("Package", no_wrap=True)
    table.add_column("Local Version")
    table.add_column("Remote Version")
    table.add_column("Status")

    local_packages = local_env.get("packages", {})
    remote_packages = remote_env.get("packages", {})

    # Only show packages that exist on the server
    server_packages = set(remote_packages.keys())

    # Sort with critical packages first, then alphabetically
    def sort_key(pkg):
        is_critical = pkg.lower() in CRITICAL_PACKAGES
        return (not is_critical, pkg.lower())

    for pkg in sorted(server_packages, key=sort_key):
        local_ver = local_packages.get(pkg, "-")
        remote_ver = remote_packages.get(pkg, "-")

        if local_ver == remote_ver:
            color = "dim"
            status = "✓"
        elif pkg.lower() in CRITICAL_PACKAGES:
            color = "red"
            status = "⚠ CRITICAL"
        else:
            color = "yellow"
            status = "≠"

        if _SUPPORTS_COLOR:
            table.add_row(
                f"[{color}]{pkg}[/]",
                f"[{color}]{local_ver}[/]",
                f"[{color}]{remote_ver}[/]",
                f"[{color}]{status}[/]",
            )
        else:
            table.add_row(
                pkg,
                local_ver,
                remote_ver,
                status,
            )

    return table


def compare() -> None:
    """
    Compare the local Python environment with the NDIF remote environment and print the results.

    Returns:
        None
    """
    local_env = get_local_env()
    remote_env = get_remote_env()

    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=_SUPPORTS_COLOR,
        color_system="auto" if _SUPPORTS_COLOR else None,
        stderr=False,
    )

    # Python version comparison
    local_py = local_env.get("python_version", "Unknown").split()[0]
    remote_py = remote_env.get("python_version", "Unknown").split()[0]

    if local_py == remote_py:
        py_color = "dim"
        py_status = "✓"
    else:
        py_color = "yellow"
        py_status = "≠"

    if _SUPPORTS_COLOR:
        console.print(f"[bold]Python Version:[/]")
        console.print(f"  [{py_color}]Local:  {local_py}[/]")
        console.print(f"  [{py_color}]Remote: {remote_py} {py_status}[/]")
    else:
        console.print("Python Version:")
        console.print(f"  Local:  {local_py}")
        console.print(f"  Remote: {remote_py} {py_status}")
    console.print()
    table = build_table(local_env, remote_env)
    console.print(table)
    print(buf.getvalue())
