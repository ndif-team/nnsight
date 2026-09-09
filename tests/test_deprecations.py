"""Every deprecation nnsight declares, and the property that makes them useful.

A deprecation warning is only worth the line it costs if the person running the
deprecated code sees it. Python's default filters show a ``DeprecationWarning``
only when it is raised from ``__main__``, so the same call warns in a script and
warns to nobody once it moves into a helper module, a package, or a library —
which is where most code being ported lives. `NNsightDeprecationWarning` is a
``FutureWarning`` for exactly that reason.

That property cannot be tested from inside pytest, which installs its own
filters: a test that sets a filter proves only that the warning was raised, never
that a user would have seen it. `TestVisibleUnderDefaultFilters` therefore runs a
subprocess with untouched filters and reads its stderr.
"""

import subprocess
import sys
import textwrap
import warnings
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn

import nnsight
from nnsight import NNsightDeprecationWarning

PROMPT = "The Eiffel Tower is in the city of"
VLM_REPO = "trl-internal-testing/tiny-LlavaForConditionalGeneration"


@pytest.fixture(scope="module")
def envoy():
    return nnsight.NNsight(nn.Linear(8, 8))


@contextmanager
def no_deprecation():
    """Fail if anything inside raises an `NNsightDeprecationWarning`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", NNsightDeprecationWarning)
        yield


class TestCategory:
    def test_is_a_future_warning(self):
        # FutureWarning is the category Python reserves for deprecations aimed at
        # a library's users, and the only one of the two shown by default outside
        # __main__. Subclassing DeprecationWarning instead would put every nnsight
        # deprecation back behind the `ignore::DeprecationWarning` default filter.
        assert issubclass(NNsightDeprecationWarning, FutureWarning)
        assert not issubclass(NNsightDeprecationWarning, DeprecationWarning)

    def test_exported(self):
        # Users silence nnsight's deprecations by naming this class; it has to be
        # reachable as `nnsight.NNsightDeprecationWarning`.
        assert nnsight.NNsightDeprecationWarning is NNsightDeprecationWarning
        assert "NNsightDeprecationWarning" in nnsight.__all__


class TestInventory:
    """One test per deprecation nnsight declares. Each names its replacement."""

    def test_language_model(self):
        with pytest.warns(NNsightDeprecationWarning, match=r"TransformersModel"):
            nnsight.LanguageModel("openai-community/gpt2")

    def test_vision_language_model(self):
        pytest.importorskip("PIL")
        with pytest.warns(NNsightDeprecationWarning, match=r"image-text-to-text"):
            nnsight.VisionLanguageModel(VLM_REPO)

    def test_model_iter(self, envoy):
        with pytest.warns(NNsightDeprecationWarning, match=r"use tracer\.iter"):
            envoy.iter

    def test_model_all(self, envoy):
        with pytest.warns(NNsightDeprecationWarning, match=r"use tracer\.all\(\)"):
            envoy.all()

    def test_with_iter_block(self, envoy):
        with pytest.warns(NNsightDeprecationWarning, match=r"for step in tracer\.iter"):
            with envoy.trace(torch.randn(1, 8)) as tracer:
                with tracer.iter[:1]:
                    nnsight.save(envoy.output)

    @torch.no_grad()
    def test_generator_output(self, gpt2):
        with pytest.warns(NNsightDeprecationWarning, match=r"use tracer\.result"):
            with gpt2.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
                nnsight.save(gpt2.generator.output)

    @torch.no_grad()
    def test_generator_output_write(self, gpt2):
        with pytest.warns(NNsightDeprecationWarning, match=r"use tracer\.result"):
            with gpt2.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
                gpt2.generator.output = torch.zeros(1, 1, dtype=torch.long)

    def test_ndif_status(self, monkeypatch):
        monkeypatch.setattr(nnsight.CONFIG.API, "HOST", "http://localhost:1")
        with pytest.warns(NNsightDeprecationWarning, match=r"use nnsight\.status\(\)"):
            nnsight.ndif_status()


class TestNotDeprecated:
    """The replacements themselves warn about nothing."""

    def test_for_form_and_current_accessors(self, envoy):
        with no_deprecation():
            with envoy.trace(torch.randn(1, 8)) as tracer:
                for _ in tracer.iter[:1]:
                    nnsight.save(envoy.output)

    @torch.no_grad()
    def test_streamer_output(self, gpt2):
        # Only the generator's own `.output` is deprecated; the per-step tokens
        # under it are the reason the module still exists.
        with no_deprecation():
            with gpt2.generate(PROMPT, max_new_tokens=2, do_sample=False) as tracer:
                for _ in tracer.iter[:2]:
                    nnsight.save(gpt2.generator.streamer.output)


# Run from an imported module, never from __main__ — the case Python's default
# DeprecationWarning filters hide, and the whole point of the FutureWarning
# category. `ported` stands in for the package a porting user's code lives in.
_PORTED = """
import warnings

import torch
import nnsight


def run():
    model = nnsight.NNsight(torch.nn.Linear(8, 8))
    model.iter
    model.all()
    with model.trace(torch.randn(1, 8)) as tracer:
        with tracer.iter[:1]:
            nnsight.save(model.output)
    nnsight.CONFIG.API.HOST = "http://localhost:1"
    try:
        nnsight.ndif_status()
    except Exception:
        pass


def someone_elses_deprecation():
    warnings.warn("A NEIGHBOURING LIBRARY'S DEPRECATION", DeprecationWarning)
"""

_DRIVER = """
import sys

import nnsight

sys.path.insert(0, sys.argv[1])
import ported

ported.run()
ported.someone_elses_deprecation()
"""


@pytest.fixture(scope="module")
def ported_run(tmp_path_factory):
    """Run the deprecated idioms in a fresh interpreter, from an imported module."""
    directory = tmp_path_factory.mktemp("ported")
    (directory / "ported.py").write_text(textwrap.dedent(_PORTED))
    (directory / "driver.py").write_text(textwrap.dedent(_DRIVER))
    return subprocess.run(
        [sys.executable, str(directory / "driver.py"), str(directory)],
        capture_output=True,
        text=True,
        timeout=900,
    )


class TestVisibleUnderDefaultFilters:
    """The deprecations reach a user whose call site is not ``__main__``.

    A subprocess with untouched warning filters is the only honest test of this:
    pytest installs ``always::DeprecationWarning`` for every test it runs, so an
    in-process assertion passes just as happily against a warning no user sees.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "model.iter is deprecated",
            "model.all() is deprecated",
            "block form is deprecated",
            "ndif_status() is deprecated",
        ],
    )
    def test_warning_reaches_stderr(self, ported_run, message):
        assert ported_run.returncode == 0, ported_run.stderr
        assert message in ported_run.stderr

    def test_points_at_the_callers_line(self, ported_run):
        # The `with` form warns from inside the tracer machinery, so the location
        # comes from the captured block rather than a stack level; a warning
        # naming iterator.py names no block for the reader to rewrite.
        lines = textwrap.dedent(_PORTED).splitlines()
        lineno = lines.index("        with tracer.iter[:1]:") + 1
        assert f"ported.py:{lineno}:" in ported_run.stderr

    def test_leaves_other_libraries_deprecations_hidden(self, ported_run):
        # Silence is bad, but a library that widens the global filters to fix it
        # is worse: it overrides the -W flags and PYTHONWARNINGS its user chose,
        # and speaks for every other library in the process. nnsight registers no
        # filters, so a neighbour's DeprecationWarning stays exactly as hidden as
        # Python left it.
        assert "NEIGHBOURING LIBRARY" not in ported_run.stderr
