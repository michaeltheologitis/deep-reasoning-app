"""Chats: each one's id, title, when it was made, and what its next message runs on and in,
in the app file beside the Library, so a copy of the file carries them.

A chat's turns are its runs, whose events runs.py keeps in the same file.
"""

import os
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from coolname import generate_slug

from deep_reasoning.library import RUN_ID, RUN_ID_FORMAT, LibraryError, open_library

log = structlog.get_logger()

# A chat id: plain characters only, like the ids the app makes.
CHAT_ID = r"^[A-Za-z0-9_-]{1,100}$"
CHOICES = ("namespace", "provider", "model", "folder")
COLUMNS = "id, title, created, namespace, provider, model, folder"
ADD_CHAT = (
    "INSERT INTO chats (id, title, created, folder)"
    " VALUES (:id, :title, :created, :folder)"
)
NEW_TITLE = "New chat"
# The longest title kept: a longer one, typed or taken from a first message, is cut.
TITLE_CHARS = 200
RUN_IDS = "SELECT chat, id FROM runs WHERE typeof(id) = 'text'"
# Three words, like "tunneling-easygoing-mosquito": over 300 million ids.
ID_WORDS = 3
# A new id is tried this many times before the chat is refused; a taken one is rare.
ID_TRIES = 10
ID_TAKEN = "SQLITE_CONSTRAINT_PRIMARYKEY"
NO_FREE_ID = f"The Library file had no free chat id in {ID_TRIES} tries; try again."

Chat = dict[str, Any]


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """The app file at *path*, laid out if new. Autocommit: every write here is one
    statement, which SQLite runs as a transaction of its own."""
    with closing(open_library(path)) as connection:
        connection.row_factory = sqlite3.Row
        yield connection


def well_formed(chat: sqlite3.Row) -> bool:
    """Whether the app could have written the chat's row, as a file from someone else may
    not have: its id names its scratch folder, and its folder is where its runs work."""
    chat_id, folder = chat["id"], chat["folder"]
    plain_id = isinstance(chat_id, str) and re.fullmatch(CHAT_ID, chat_id)
    absolute = folder is None or (isinstance(folder, str) and os.path.isabs(folder))
    if plain_id and absolute:
        return True
    log.warning("chats.left_out", id=repr(chat_id), folder=repr(folder))
    return False


def chats_in(connection: sqlite3.Connection) -> list[Chat]:
    """Every chat in the open app file that the app could have written, newest first."""
    cursor = connection.cursor()
    cursor.row_factory = sqlite3.Row
    rows = cursor.execute(
        f"SELECT {COLUMNS} FROM chats ORDER BY created DESC, rowid DESC"
    )
    return [dict(row) for row in rows if well_formed(row)]


def latest_runs(connection: sqlite3.Connection, chat: str | None = None) -> dict:
    """Each chat's latest run, or *chat*'s, by its id; an id the app couldn't have made,
    as a file from someone else may hold, is left out."""
    query, args = (
        (RUN_IDS, ()) if chat is None else (f"{RUN_IDS} AND chat = ?", (chat,))
    )
    latest = {}
    for owner, run in connection.execute(query, args):
        if re.fullmatch(RUN_ID, run) and run > latest.get(owner, ""):
            latest[owner] = run
    return latest


def last_used(chat: Chat) -> float:
    """When the chat was last used: its latest run's start, else when it was made."""
    created = chat["created"]
    made = created if isinstance(created, int | float) else 0.0
    if chat["last_run"] is None:
        return made
    try:
        began = datetime.strptime(chat["last_run"], RUN_ID_FORMAT)
    except ValueError:  # the shape of an id, but no date, as month 13
        return made
    return began.replace(tzinfo=UTC).timestamp()


def list_chats(path: Path) -> list[Chat]:
    """Every chat the app could have written, with its latest run, the one used last
    first."""
    with connect(path) as connection:
        chats, latest = chats_in(connection), latest_runs(connection)
    chats = [chat | {"last_run": latest.get(chat["id"])} for chat in chats]
    return sorted(chats, key=last_used, reverse=True)


def find_chat(path: Path, chat: str) -> Chat | None:
    """The chat, with its latest run; None when it isn't there."""
    with connect(path) as connection:
        row = connection.execute(
            f"SELECT {COLUMNS} FROM chats WHERE id = ?", (chat,)
        ).fetchone()
        if row is None or not well_formed(row):
            return None
        return dict(row) | {"last_run": latest_runs(connection, chat).get(chat)}


def create_chat(path: Path, folder: str | None = None) -> Chat:
    """A new chat titled "New chat", under a readable id no other chat has, working in
    *folder*, or in its scratch folder when None."""
    chat = {
        "id": generate_slug(ID_WORDS),
        "title": NEW_TITLE,
        "created": time.time(),
        "folder": folder,
    }
    with connect(path) as connection:
        for _ in range(ID_TRIES):
            if inserted(connection, chat):
                return dict.fromkeys(CHOICES) | chat | {"last_run": None}
            chat["id"] = generate_slug(ID_WORDS)
    raise LibraryError(NO_FREE_ID)


def inserted(connection: sqlite3.Connection, chat: Chat) -> bool:
    """Whether *chat* went in: False when its id is taken. Any other refusal, as a shared
    file's own constraints may make, is a LibraryError."""
    try:
        connection.execute(ADD_CHAT, chat)
    except sqlite3.IntegrityError as error:
        if error.sqlite_errorname == ID_TAKEN:
            return False
        raise LibraryError(
            f"The Library file refused the new chat: {error}."
        ) from error
    return True


def capped(title: str) -> str:
    return title[:TITLE_CHARS].rstrip()


def rename_chat(path: Path, chat: str, title: str) -> None:
    """Title the chat *title*, cut past TITLE_CHARS."""
    with connect(path) as connection:
        connection.execute(
            "UPDATE chats SET title = ? WHERE id = ?", (capped(title), chat)
        )


def choose(
    path: Path,
    chat: str,
    namespace: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Set what the chat's next message runs on; a choice left None stays as it was."""
    with connect(path) as connection:
        connection.execute(
            "UPDATE chats SET namespace = coalesce(?, namespace),"
            " provider = coalesce(?, provider), model = coalesce(?, model) WHERE id = ?",
            (namespace, provider, model, chat),
        )


def choose_folder(path: Path, chat: str, folder: str | None) -> None:
    """Set the folder the chat's next message works in; None for its scratch folder."""
    with connect(path) as connection:
        connection.execute("UPDATE chats SET folder = ? WHERE id = ?", (folder, chat))


def title_new_chat(path: Path, chat: str, message: str) -> str | None:
    """Title a chat still called "New chat" after *message*'s first line that isn't
    blank, cut past TITLE_CHARS; the title, or None when the chat keeps its own."""
    title = capped(message.strip().partition("\n")[0].strip())
    if not title:
        return None
    with connect(path) as connection:
        titled = connection.execute(
            "UPDATE chats SET title = ? WHERE id = ? AND title = ?",
            (title, chat, NEW_TITLE),
        ).rowcount
    return title if titled else None


def delete_chat(path: Path, chat: str) -> None:
    with connect(path) as connection:
        connection.execute("DELETE FROM chats WHERE id = ?", (chat,))
