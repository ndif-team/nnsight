import json
import os
from typing import Optional, Union

from huggingface_hub import HfApi, constants
from huggingface_hub.file_download import repo_folder_name
from typing_extensions import Self

from .mixins import RemoteableMixin

ID_CACHE = {}


class HuggingFaceModel(RemoteableMixin):
    """Base class for NNsight wrappers around HuggingFace Hub models.

    Adds HuggingFace repository handling (repo ID, revision) and
    persistent edit export/import on top of :class:`RemoteableMixin`.

    Args:
        repo_id (str): HuggingFace repository ID (e.g. ``"openai-community/gpt2"``)
            or a pre-loaded ``torch.nn.Module``.
        *args: Forwarded to the parent mixin chain.
        revision (Optional[str]): Git revision (branch, tag, or commit hash)
            of the model repository. Defaults to ``None`` (latest).
        import_edits (Union[bool, str]): If ``True``, import previously
            exported edits using the default variant. If a string, use
            it as the variant name. Defaults to ``False``.
        **kwargs: Forwarded to the parent mixin chain and ultimately to
            the model loading function.

    Attributes:
        repo_id (str): The HuggingFace repository ID.
        revision (Optional[str]): The repository revision.
    """

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
        """Export persistent model edits to disk.

        Edits created via ``model.edit(inplace=True)`` are serialized
        and saved so they can be reloaded later with :meth:`import_edits`.

        Args:
            name (Optional[str]): Export name. Defaults to a name
                derived from the HuggingFace repo ID.
            export_dir (Optional[str]): Directory to save exports.
                Defaults to the HuggingFace cache under ``nnsight/exports``.
            variant (str): Named variant for this set of edits.
                Defaults to ``'__default__'``.
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
        """Import previously exported model edits from disk.

        Loads edits that were saved with :meth:`export_edits` and
        applies them as persistent in-place edits on this model.

        Args:
            name (Optional[str]): Export name. Defaults to a name
                derived from the HuggingFace repo ID.
            export_dir (Optional[str]): Directory to load exports from.
                Defaults to the HuggingFace cache under ``nnsight/exports``.
            variant (str): Named variant to load.
                Defaults to ``'__default__'``.
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

        if self.repo_id not in ID_CACHE:
            ID_CACHE[self.repo_id] = HfApi().model_info(self.repo_id).id

        repo_id = ID_CACHE[self.repo_id]

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
