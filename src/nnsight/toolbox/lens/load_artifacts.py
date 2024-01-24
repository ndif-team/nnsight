"""Load lens artifacts from the hub or locally storage.

See https://github.com/AlignmentResearch/tuned-lens for more.
"""
import os
from pathlib import Path
from typing import Optional
from typing import Set, Tuple

from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


def available_lens_artifacts(
    repo_id: str,
    repo_type: str,
    revision: str = "main",
    config_file: str = "config.json",
    ckpt_file: str = "params.pt",
    subfolder: str = "lens",
) -> Set[str]:
    """Get the available lens artifacts from the hub."""
    fs = HfFileSystem()

    repo_type = repo_type + "s" if not repo_type.endswith("s") else repo_type

    root = Path(repo_type, repo_id, subfolder)
    with_config = map(
        Path,
        fs.glob(
            (root / "**" / config_file).as_posix(), revision=revision  # type: ignore
        ),
    )

    with_pt = map(
        Path,
        fs.glob(
            (root / "**" / ckpt_file).as_posix(), revision=revision  # type: ignore
        ),
    )
    paths = {p.parent for p in with_pt}.intersection({p.parent for p in with_config})
    return {p.relative_to(root).as_posix() for p in paths}


def load_lens_artifacts(
    resource_id: str,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: str = "main",
    config_file: str = "config.json",
    ckpt_file: str = "params.pt",
    subfolder: str = "lens",
    cache_dir: Optional[str] = None,
) -> Tuple[Path, Path]:
    """First checks for lens resource locally then tries to download it from the hub.

    Args:
        resource_id: The id of the lens resource.
        repo_id: The repository to download the lens from. Defaults to
            'AlignmentResearch/tuned-lens'. However, this default can be overridden by
            setting the TUNED_LENS_REPO_ID environment variable.
        repo_type: The type of repository to download the lens from. Defaults to
            'space'. However, this default can be overridden by setting the
            TUNED_LENS_REPO_TYPE environment variable.
        config_file: The name of the config file in the folder contain the lens.
        ckpt_file: The name of the checkpoint file in the folder contain the lens.
        revision: The revision of the lens to download.
        subfolder: The subfolder of the repository to download the lens from.
        cache_dir: The directory to cache the lens in.

    Returns:
        * The path to the config.json file
        * The path to the params.pt file

    Raises:
        ValueError: if the lens resource could not be found.
    """
    if repo_id is None:
        if os.environ.get("TUNED_LENS_REPO_ID"):
            repo_id = os.environ["TUNED_LENS_REPO_ID"]
        else:
            repo_id = "AlignmentResearch/tuned-lens"

    if repo_type is None:
        if os.environ.get("TUNED_LENS_REPO_TYPE"):
            repo_type = os.environ["TUNED_LENS_REPO_TYPE"]
        else:
            repo_type = "space"

    # Fist check if the resource id is a path to a folder that exists
    local_path = Path(resource_id)
    if (local_path / config_file).exists() and (local_path / ckpt_file).exists():
        return local_path / config_file, local_path / ckpt_file

    resource_folder = "/".join((subfolder, resource_id))
    try:
        params_path = hf_hub_download(
            filename=ckpt_file,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            subfolder=resource_folder,
            cache_dir=cache_dir,
        )

        config_path = hf_hub_download(
            filename=config_file,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            subfolder=resource_folder,
            cache_dir=cache_dir,
        )
    except EntryNotFoundError:
        available_lenses = available_lens_artifacts(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            config_file=config_file,
            ckpt_file=ckpt_file,
            subfolder=subfolder,
        )
        message = (
            f"Could not find lens at the specified resource id. Available lens"
            f"resources are: {', '.join(available_lenses)}"
        )
        raise ValueError(message)

    if config_path is not None and params_path is not None:
        return Path(config_path), Path(params_path)

    raise ValueError("Could not find lens resource locally or on the hf hub.")