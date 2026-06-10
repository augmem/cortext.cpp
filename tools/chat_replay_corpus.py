"""Shared helpers for locating a chat-replay corpus transcript.

A chat-replay corpus is a directory containing exactly one top-level
``.txt`` transcript plus optional media files. The transcript is a plain
text export where each message starts with a header line of the form

    YYYY-MM-DD HH:MM:SS <direction marker containing " to " or " from ">

followed by the message body on subsequent lines, with messages separated
by a long dash rule. ``" from "`` in the header marks a message from the
contact; anything else is from the user.
"""

from __future__ import annotations

import pathlib


def discover_transcript(
    input_dir: pathlib.Path | str,
    override: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Return the transcript path for a corpus directory.

    ``override`` (e.g. from a ``--transcript`` flag) wins when given.
    Otherwise the directory must contain exactly one top-level ``.txt``
    file; when several are present, a single legacy ``Messages - *.txt``
    export is accepted as the transcript.
    """
    if override:
        path = pathlib.Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"transcript not found: {path}")
        return path
    directory = pathlib.Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"input dir not found or not a directory: {directory}; expected a "
            "chat export directory with one top-level .txt transcript plus "
            "optional media files"
        )
    texts = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix == ".txt"
    )
    if len(texts) == 1:
        return texts[0]
    legacy = [p for p in texts if p.name.startswith("Messages - ")]
    if len(legacy) == 1:
        return legacy[0]
    if not texts:
        raise FileNotFoundError(
            f"no .txt transcript found in {directory}; expected a chat export "
            'with timestamped "YYYY-MM-DD HH:MM:SS ... to|from ..." message '
            "headers"
        )
    raise FileNotFoundError(
        f"multiple .txt transcripts in {directory}: "
        f"{', '.join(p.name for p in texts)}; pass --transcript to choose one"
    )
