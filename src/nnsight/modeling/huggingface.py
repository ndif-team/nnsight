import json
import os
from typing import Optional, Union

from huggingface_hub import HfApi, constants
from huggingface_hub.file_download import repo_folder_name
from typing_extensions import Self

from .mixins import RemoteableMixin


class HuggingFaceModel(RemoteableMixin):

    def __init__(
        self,
        repo_id: str,
        *args,
        revision: Optional[str] = None,
        import_edits: Union[bool, str] = False,
        **kwargs,
    ):

        self.repo_id = (
            repo_id
            if isinstance(repo_id, str)
            else getattr(repo_id, "name_or_path", None)
        )
        self.revision = revision

        super().__init__(repo_id, *args, revision=revision, **kwargs)

        if import_edits:

            if isinstance(import_edits, str):

                self.import_edits(variant=import_edits)

            else:

                self.import_edits()

    def export_edits(
        self,
        name: Optional[str] = None,
        export_dir: Optional[str] = None,
        variant: str = "__default__",
    ):
        """TODO

        Args:
            name (Optional[str], optional): _description_. Defaults to None.
            export_dir (Optional[str], optional): _description_. Defaults to None.
            variant (str, optional): _description_. Defaults to '__default__'.
        """

        if name is None:
            name = repo_folder_name(repo_id=self.repo_id, repo_type="model")

            if export_dir is None:
                export_dir = os.path.join(
                    constants.HF_HUB_CACHE, name, "nnsight", "exports"
                )
                name = ""

        super().export_edits(name, export_dir=export_dir, variant=variant)

    def import_edits(
        self,
        name: Optional[str] = None,
        export_dir: Optional[str] = None,
        variant: str = "__default__",
    ):
        """TODO

        Args:
            name (Optional[str], optional): _description_. Defaults to None.
            export_dir (Optional[str], optional): _description_. Defaults to None.
            variant (str, optional): _description_. Defaults to '__default__'.
        """

        if name is None:
            name = repo_folder_name(repo_id=self.repo_id, repo_type="model")

            if export_dir is None:
                export_dir = os.path.join(
                    constants.HF_HUB_CACHE, name, "nnsight", "exports"
                )
                name = ""

        super().import_edits(name, export_dir=export_dir, variant=variant)

    def _remoteable_model_key(self) -> str:

        repo_id = HfApi().model_info(self.repo_id).id

        return json.dumps(
            {
                "repo_id": repo_id,
                "revision": self.revision,
            }  # , "torch_dtype": str(self._model.dtype)}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs) -> Self:

        kwargs = {**json.loads(model_key), **kwargs}

        repo_id = kwargs.pop("repo_id")

        revision = kwargs.pop("revision", "main")

        return cls(repo_id, revision=revision, **kwargs)
