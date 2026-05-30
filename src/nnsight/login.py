"""Login helpers for storing your NDIF API key.

This module exposes two entry points:

- ``nnsight.login()`` for use in a notebook or interactive session. It prompts
  for the API key with ``getpass`` so the key is never echoed to the screen and
  saves it through the existing config machinery.
- ``main()`` for the ``nnsight login`` console command, which does the same thing
  from a terminal.

Both persist the key by calling ``CONFIG.set_default_api_key``, which writes it
to ``config.yaml`` so future sessions pick it up automatically.
"""

from getpass import getpass
from typing import Optional


def login(api_key: Optional[str] = None) -> None:
    """Store your NDIF API key so future sessions can use it.

    Args:
        api_key: The NDIF API key. If not given, you will be prompted for it.
            The prompt uses ``getpass`` so the key is not echoed.

    Examples:
        >>> from nnsight import login
        >>> login()
        Enter your NDIF API key:
        NDIF API key saved.
    """

    from . import CONFIG

    if api_key is None:
        api_key = getpass("Enter your NDIF API key: ")

    api_key = api_key.strip()

    if not api_key:
        print("No API key provided. Nothing was saved.")
        return

    CONFIG.set_default_api_key(api_key)

    print("NDIF API key saved.")


def main() -> None:
    """Console entry point for the ``nnsight login`` command."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="nnsight",
        description="Command line tools for nnsight.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("login", help="Store your NDIF API key.")

    args = parser.parse_args()

    if args.command == "login":
        login()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
