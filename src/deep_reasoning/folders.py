"""Working folders: where each chat's Python runs and reads and writes files.

A chat works in the folder its user chose, or else in its scratch folder, work/<chat>/ in
the app file's data folder, made on its first run and deleted with the chat. A folder the
user chose is never deleted. Paths reach the page with your home folder written as ~.
"""

import hashlib
import itertools
import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jinja2
import structlog

from deep_reasoning.chats import CHAT_ID

log = structlog.get_logger()

WORK = "work"
FILES = "files"
# Of the sha256 of the app file's path: 64 bits, so no two files on a computer share one.
PATH_DIGITS = 16
HOME = "~"
GONE = "That folder no longer exists — pick another."
# What every agent of a run is told of the chat's folder: so many of its names at most,
# from so many entries read, so a huge folder costs nothing.
SHOWN_NAMES = 40
READ_ENTRIES = 1_000
FOLDER_LINE = jinja2.Template(
    "This chat's working folder is {{ path }}, {% if scratch %}the chat's own scratch"
    " folder, not one of the user's{% else %}the user's own{% endif %}{% if not read %}."
    "{% elif names %}, holding (a / marks a folder): {{ names | join(', ') }}"
    "{% if more %}, and {% if cut %}over {% endif %}{{ more }} more{% endif %}."
    "{% else %}: it is empty.{% endif %}",
    undefined=jinja2.StrictUndefined,
)
# Told last to each agent: told it earlier, or not at all, agents asked for an upload.
PICK_LINE = jinja2.Template(
    "There is no upload: for files that aren't in this chat's folder, ask the user to"
    " click {{ button }} under the message box and pick the folder holding them.",
    undefined=jinja2.StrictUndefined,
)
# The folder button under the message box shows the last part of the folder's path, ~ for
# your home folder, cut to so many characters with an ellipsis, as web/app.js and clip() in
# web/fold.js write it; this for a scratch folder.
FOLDER_NAME_CHARS = 24
ELLIPSIS = "…"
SCRATCH = "scratch"


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


def no_folder(folder: Path) -> str:
    return f"There is no folder {short_path(folder)}."


def existing_folder(typed: str) -> Path:
    """The folder at *typed*, or why there is none."""
    return checked_folder(typed_path(typed))


def checked_folder(folder: Path) -> Path:
    """*folder*, made absolute, if it is a folder, or why not. os.path's checks answer
    False for any path the system can't take (too long, a NUL in it) instead of raising."""
    if os.path.isdir(folder):
        return folder
    if os.path.exists(folder):
        raise FolderError(f"{short_path(folder)} is a file, not a folder.")
    raise FolderError(no_folder(folder))


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
    None at the root. Nothing at *typed*, as with a path half typed, lists nothing and
    says so (problem), as the picker asks at each pause in typing: no refusal."""
    folder = typed_path(typed)
    if not os.path.exists(folder):
        return {
            "path": short_path(folder),
            "parent": None,
            "folders": [],
            "problem": no_folder(folder),
        }
    checked_folder(folder)
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
        "problem": None,
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


def folder_lines(folder: Path, scratch: bool) -> tuple[str, str]:
    """What a run tells its agents of the chat's working *folder*, its *scratch* folder
    or one the user picked: where it is and the names in it, which every agent is told;
    and how the user picks another, which every agent the engine gives its namespace's
    instructions is told last."""
    found = listing(folder)
    names, whole = found or ([], True)
    shown = [python_text(name) for name in names[:SHOWN_NAMES]]
    where = FOLDER_LINE.render(
        path=python_text(str(folder)),
        scratch=scratch,
        read=found is not None,
        names=shown,
        more=len(names) - len(shown),
        cut=not whole,
    )
    return where, PICK_LINE.render(button=python_text(button_name(folder, scratch)))


def button_name(folder: Path, scratch: bool) -> str:
    """What the folder button under the message box shows for the chat's *folder*."""
    if scratch:
        return SCRATCH
    name = short_path(folder).split("/")[-1] or "/"
    if len(name) <= FOLDER_NAME_CHARS:
        return name
    return name[: FOLDER_NAME_CHARS - 1] + ELLIPSIS


def listing(folder: Path) -> tuple[list[str], bool] | None:
    """The names in *folder* but hidden ones, sorted, a folder's ending in /, from its
    first READ_ENTRIES entries, and whether those were all; None when it can't be read."""
    try:
        with os.scandir(folder) as entries:
            read = list(itertools.islice(entries, READ_ENTRIES + 1))
    except OSError:
        return None
    names = [
        marked(entry) for entry in read[:READ_ENTRIES] if not entry.name.startswith(".")
    ]
    return sorted(names, key=str.lower), len(read) <= READ_ENTRIES


def marked(entry: os.DirEntry) -> str:
    """*entry*'s name, with a / after a folder's."""
    try:
        is_folder = entry.is_dir()
    except OSError:
        is_folder = False
    return f"{entry.name}/" if is_folder else entry.name


def python_text(text: str) -> str:
    """*text* as Python writes it, quoted, so a name from the user's disk reads whole
    whatever it holds (a newline, a quote, bytes that aren't UTF-8) and copies into code
    as it is; each { written \\x7b, as the engine renders the suffix it goes in as a
    template."""
    return repr(text).replace("{", "\\x7b")


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
