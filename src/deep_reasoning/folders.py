"""Working folders: where each chat's Python runs and reads and writes files.

A chat works in the folder its user chose, or else in its scratch folder, work/<chat>/ in
the app file's data folder, made on its first run and deleted with the chat. A folder the
user chose is never deleted. Paths reach the page with your home folder written as ~.
"""

import hashlib
import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import structlog

from deep_reasoning.chats import CHAT_ID

log = structlog.get_logger()

WORK = "work"
FILES = "files"
# Of the sha256 of the app file's path: 64 bits, so no two files on a computer share one.
PATH_DIGITS = 16
HOME = "~"
GONE = "That folder no longer exists — pick another."


class FolderError(ValueError):
    """A folder the app can't use, in words for the person who asked."""


def short_path(path: Path) -> str:
    """*path* with your home folder written as ~."""
    home = Path.home()
    if not path.is_relative_to(home):
        return str(path)
    rest = path.relative_to(home)
    return HOME if rest == Path() else f"{HOME}/{rest}"


def typed_path(typed: str) -> Path:
    """A typed path made absolute: ~ expanded, and a relative one taken from your home
    folder, where the folder browser starts."""
    return Path(os.path.normpath(Path.home() / os.path.expanduser(typed.strip())))


def existing_folder(typed: str) -> Path:
    """The folder at *typed*, or why there is none. os.path's checks answer False for any
    path the system can't take (too long, a NUL in it) instead of raising."""
    folder = typed_path(typed)
    if os.path.isdir(folder):
        return folder
    if os.path.exists(folder):
        raise FolderError(f"{short_path(folder)} is a file, not a folder.")
    raise FolderError(f"There is no folder {short_path(folder)}.")


def chosen_folder(typed: str | None) -> str | None:
    """The folder a chat or the settings chose, as kept; None, for a scratch folder, when
    blank."""
    if typed is None or not typed.strip():
        return None
    return str(existing_folder(typed))


def listable(entry: os.DirEntry) -> bool:
    """A folder the page can show and name back: not one whose name isn't UTF-8, nor
    one the system can't look at (a symlink loop, a link into a folder you can't open)."""
    try:
        entry.name.encode()
        return entry.is_dir()
    except (UnicodeEncodeError, OSError):
        return False


def list_folders(typed: str) -> dict[str, Any]:
    """The folder at *typed* and its subfolders, hidden ones last; its parent for "..",
    None at the root."""
    folder = existing_folder(typed)
    try:
        with os.scandir(folder) as entries:
            names = [entry.name for entry in entries if listable(entry)]
    except OSError as error:
        raise FolderError(
            f"Can't open {short_path(folder)}: {error.strerror or error}."
        ) from error
    names.sort(key=lambda name: (name.startswith("."), name.lower()))
    parent = None if folder.parent == folder else short_path(folder.parent)
    return {
        "path": short_path(folder),
        "parent": parent,
        "folders": [
            {"name": name, "path": short_path(folder / name)} for name in names
        ],
    }


def data_folder(home: Path, library: Path) -> Path:
    """Where the run and scratch folders of the app file at *library* live: a folder of
    the app's *home* named for the file's real path, so a copy of the file never shares
    them with the original, and a moved file starts afresh, deleting nothing."""
    digest = hashlib.sha256(str(library.resolve()).encode()).hexdigest()
    return home / FILES / digest[:PATH_DIGITS]


def scratch_folder(data: Path, chat: str) -> Path:
    """work/<chat> in the app file's *data* folder. Only a plain id names one, so no id
    reaches outside."""
    if not re.fullmatch(CHAT_ID, chat):
        raise ValueError(f"{chat!r} is not a chat id")
    return data / WORK / chat


def starting_folder(default: str | None) -> str | None:
    """Where a new chat works: the default folder while it is there; else, as when the app
    file came from another computer, its scratch folder (None)."""
    if default is None or not os.path.isdir(default):
        return None
    return default


def folder_gone(folder: str | None) -> bool:
    """Whether the *folder* a chat chose is no longer there, so its next run is refused."""
    return folder is not None and not os.path.isdir(folder)


def working_folder(data: Path, chat: dict[str, Any]) -> Path:
    """Where the chat's next run works: the folder it chose, which must still be there, or
    its scratch folder, made if new."""
    if folder_gone(chat["folder"]):
        raise FolderError(GONE)
    if chat["folder"] is not None:
        return Path(chat["folder"])
    scratch = scratch_folder(data, chat["id"])
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def holds_files(folder: Path) -> bool:
    """Whether *folder* is there and not empty, as a scratch folder an agent wrote in."""
    try:
        return any(folder.iterdir())
    except OSError:
        return False


def holds_any(folder: Path, folders: Iterable[str]) -> bool:
    """Whether *folder* is or holds one of *folders*, each where it really is."""
    # realpath, as Path.resolve raises on a symlink loop
    real = Path(os.path.realpath(folder))
    return any(Path(os.path.realpath(each)).is_relative_to(real) for each in folders)


def remove_unless_kept(folder: Path, keep: Iterable[str]) -> None:
    """Delete *folder*, one of a deleted chat's, unless it is or holds a folder in *keep*,
    the folders chosen or in use; never through a symlink. The chat is gone already, so a
    folder left behind is logged, not raised."""
    if holds_any(folder, keep):
        log.warning("folders.kept_as_chosen", folder=str(folder))
        return
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        return
    except OSError:
        log.exception("folders.not_removed", folder=str(folder))
