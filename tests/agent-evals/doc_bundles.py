"""Documentation bundles for the eval suite.

The whole point of the suite is to **benchmark the documentation**: how well
do different doc subsets help an agent succeed at nnsight tasks?

There are two evaluation modes:

- ``static``  — the chosen bundle is concatenated into the agent's system
  prompt up-front. The agent sees everything at once. Useful as a baseline
  ("what would the agent do given perfect doc recall?") but does not test
  the routing model.
- ``browse``  — the agent's system prompt is just the router (``CLAUDE.md``).
  The agent has a ``Read`` tool scoped to a list of allowed directories
  (the bundle's contents) and is expected to navigate via the router's
  links. This tests **the docs as designed**: thin router + lazy fetch.

Static-mode bundles (``DOC_BUNDLES``):

- ``minimal`` — only ``CLAUDE.md``.
- ``router`` (default) — ``CLAUDE.md`` + ``docs/concepts/`` + ``docs/gotchas/``
  + ``docs/errors/`` + ``docs/reference/``.
- ``full`` — ``CLAUDE.md`` + entire ``docs/`` tree + ``README.md`` +
  ``0.6.0.md``. (At ~870KB this exceeds the 200K standard context window;
  use it only with extended-context models or in browse mode.)
- ``legacy`` — ``CLAUDE.md`` + truncated ``NNsight.md`` (pre-``docs/``
  baseline).

Browse-mode bundles (``BROWSE_BUNDLES``):

- ``minimal`` — no ``Read``-able paths. The agent sees only the router.
- ``router`` — ``Read`` is scoped to ``docs/concepts/``, ``docs/gotchas/``,
  ``docs/errors/``, ``docs/reference/``.
- ``full`` — ``Read`` is scoped to the whole ``docs/`` tree.
"""

from __future__ import annotations

import os
from pathlib import Path


# --- bundle definitions ---------------------------------------------------

# Each bundle is a list of "selectors" — either a single file (str ending
# in ``.md``) or a folder (str without trailing slash). Folders are
# recursively included; subfolders that match ``EXCLUDE_DIRS`` are skipped.
DOC_BUNDLES: dict[str, list[str]] = {
    "minimal": [
        "CLAUDE.md",
    ],
    "router": [
        "CLAUDE.md",
        "docs/concepts",
        "docs/gotchas",
        "docs/errors",
        "docs/reference",
    ],
    "full": [
        "CLAUDE.md",
        "README.md",
        "0.6.0.md",
        "docs",
    ],
    "legacy": [
        "CLAUDE.md",
        "_LEGACY:NNsight.md",  # Special handling — see _load_legacy_nnsight_md
    ],
}


# Subdirectories to exclude when expanding folder selectors.
EXCLUDE_DIRS = {"questions", "_pycache__", "__pycache__"}

# Files to exclude (writer-question logs etc.).
EXCLUDE_FILES = {"questions.md", "followup.md", "followup-1.md"}


# Approximate context budget for the "legacy" bundle's NNsight.md slice.
# Matches the historical eval.py behavior.
LEGACY_NNSIGHT_MD_BUDGET = 50_000


# Browse-mode bundles. Selectors are all directory paths relative to
# ``nnsight_path``. The agent's system prompt is always ``CLAUDE.md`` (the
# router); these selectors become ``--add-dir`` flags on ``claude -p`` so
# the agent's ``Read`` tool can resolve files within them and only those.
BROWSE_BUNDLES: dict[str, list[str]] = {
    "minimal": [],  # no add-dirs — Read tool disabled; just the router
    "router": [
        "docs/concepts",
        "docs/gotchas",
        "docs/errors",
        "docs/reference",
    ],
    "full": [
        "docs",
    ],
    # ``legacy`` doesn't make sense in browse mode — there was no docs/ tree
    # to navigate before the migration. Use static mode if you want to
    # benchmark against the pre-docs baseline.
}


def list_doc_bundles(mode: str = "static") -> list[str]:
    """Return the list of available bundle names for the given mode."""
    if mode == "browse":
        return list(BROWSE_BUNDLES.keys())
    return list(DOC_BUNDLES.keys())


def build_browse_config(
    nnsight_path: str | Path,
    bundle: str,
) -> tuple[str, list[Path]]:
    """Return ``(system_prompt, add_dirs)`` for browse mode.

    The system prompt is always the router (``CLAUDE.md``); the agent navigates
    from there via its ``Read`` tool. ``add_dirs`` is the list of absolute
    directory paths to pass as ``--add-dir`` flags on the ``claude -p``
    invocation; these scope where the ``Read`` tool can resolve files.

    Args:
        nnsight_path: Path to the nnsight repo root.
        bundle: One of ``BROWSE_BUNDLES`` keys.

    Returns:
        Tuple of (system prompt text, list of add-dir paths).
    """
    if bundle not in BROWSE_BUNDLES:
        raise ValueError(
            f"Unknown browse bundle {bundle!r}. "
            f"Choose from: {list(BROWSE_BUNDLES.keys())}"
        )

    nnsight_path = Path(nnsight_path).resolve()
    if not nnsight_path.exists():
        raise FileNotFoundError(f"nnsight path not found: {nnsight_path}")

    claude_md = nnsight_path / "CLAUDE.md"
    if not claude_md.exists():
        raise FileNotFoundError(f"CLAUDE.md missing at {claude_md}")
    sys_prompt = claude_md.read_text()

    add_dirs: list[Path] = []
    for sel in BROWSE_BUNDLES[bundle]:
        target = nnsight_path / sel
        if target.exists() and target.is_dir():
            add_dirs.append(target)
    return sys_prompt, add_dirs


def _expand_selector(nnsight_path: Path, selector: str) -> list[Path]:
    """Expand a bundle selector to a sorted list of file paths."""
    if selector.startswith("_LEGACY:"):
        # Handled separately — never expanded as a real path.
        return []
    target = nnsight_path / selector
    if not target.exists():
        return []
    if target.is_file():
        if target.name in EXCLUDE_FILES:
            return []
        return [target]
    files: list[Path] = []
    for root, dirs, fnames in os.walk(target):
        # Prune excluded directories in place so os.walk doesn't descend.
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in fnames:
            if not fn.endswith(".md"):
                continue
            if fn in EXCLUDE_FILES:
                continue
            files.append(Path(root) / fn)
    files.sort()
    return files


def _load_legacy_nnsight_md(nnsight_path: Path) -> str:
    """Load NNsight.md truncated to LEGACY_NNSIGHT_MD_BUDGET chars."""
    p = nnsight_path / "NNsight.md"
    if not p.exists():
        return ""
    content = p.read_text()
    if len(content) > LEGACY_NNSIGHT_MD_BUDGET:
        content = content[:LEGACY_NNSIGHT_MD_BUDGET] + "\n\n[... truncated for length ...]"
    return f"# NNsight.md (Technical Details)\n\n{content}"


def build_documentation(nnsight_path: str | Path, bundle: str = "router") -> str:
    """Build a single concatenated documentation string for the chosen bundle.

    Args:
        nnsight_path: path to the nnsight repo root.
        bundle: one of ``DOC_BUNDLES`` keys.

    Returns:
        Concatenated markdown ready to feed to an agent's system prompt.
    """
    if bundle not in DOC_BUNDLES:
        raise ValueError(
            f"Unknown bundle {bundle!r}. Choose from: {list(DOC_BUNDLES.keys())}"
        )

    nnsight_path = Path(nnsight_path).resolve()
    if not nnsight_path.exists():
        raise FileNotFoundError(f"nnsight path not found: {nnsight_path}")

    selectors = DOC_BUNDLES[bundle]
    parts: list[str] = []
    seen: set[Path] = set()

    for sel in selectors:
        if sel.startswith("_LEGACY:"):
            text = _load_legacy_nnsight_md(nnsight_path)
            if text:
                parts.append(text)
            continue
        for p in _expand_selector(nnsight_path, sel):
            if p in seen:
                continue
            seen.add(p)
            try:
                rel = p.relative_to(nnsight_path)
            except ValueError:
                rel = p
            content = p.read_text()
            parts.append(f"# {rel.as_posix()}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def bundle_size_bytes(nnsight_path: str | Path, bundle: str) -> int:
    """Return the approximate size in bytes of a bundle (for budget reasoning)."""
    return len(build_documentation(nnsight_path, bundle).encode("utf-8"))


if __name__ == "__main__":
    # Quick CLI for inspecting bundles.
    import argparse
    parser = argparse.ArgumentParser(description="Inspect doc bundles.")
    parser.add_argument("--nnsight-path", default="../..")
    parser.add_argument("--bundle", choices=list(DOC_BUNDLES.keys()), default="router")
    parser.add_argument("--list-bundles", action="store_true")
    parser.add_argument("--size", action="store_true",
                        help="Print byte size for each bundle.")
    args = parser.parse_args()

    if args.list_bundles:
        for name in DOC_BUNDLES:
            print(name)
    elif args.size:
        path = Path(args.nnsight_path).resolve()
        for name in DOC_BUNDLES:
            try:
                n = bundle_size_bytes(path, name)
                print(f"{name:>10}  {n:>10,d} bytes")
            except Exception as e:
                print(f"{name:>10}  error: {e}")
    else:
        print(build_documentation(args.nnsight_path, args.bundle))
