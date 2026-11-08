"""Pages: what an agent showed the user with the app's page tool, in the app file's pages
table, each drawn in a tab of its own at /pages/<id>.

A page is a title and blocks: Markdown text; a step or a sub-agent of an earlier message,
copied from its run's record as fold.js folds it; and tables, each value as text. A shared
file is a trust boundary, so each row is checked as it is read: a row the page tool could
not have saved reads as no page, and a block it could not have saved is left out, each
with a warning.
"""

import json
import re
import sqlite3
import time
from typing import Any

import structlog
from coolname import generate_slug

from deep_reasoning.chats import ID_TAKEN, ID_TRIES, ID_WORDS
from deep_reasoning.library import LibraryError, writing
from deep_reasoning.runs import NUMBERS, UNREADABLE

log = structlog.get_logger()

PAGE_ID = r"^[A-Za-z0-9_-]{1,100}$"
NO_FREE_ID = f"The Library file had no free page id in {ID_TRIES} tries; try again."
PAGE_GONE = "The page is no longer in the Library file, so it wasn't saved again."
MAX_TITLE_CHARS = 200
MAX_BLOCKS = 200
MAX_TEXT_CHARS = 100_000
MAX_ROWS = 1_000
MAX_COLUMNS = 50
# The blocks as JSON, as the pages table holds them.
MAX_PAGE_CHARS = 2_000_000
ADD_PAGE = (
    "INSERT INTO pages VALUES (:id, :chat, :run, :node, :title, :blocks, :created)"
)
NONE = type(None)
TEXT = (str,)
MAYBE_TEXT = (str, NONE)
MAYBE_NUMBER = (int, float, NONE)
NODE = (int, str, NONE)
# What fold.js gives a sub-agent's drive and a cell, as the page draws them.
DRIVE_STATUSES = frozenset({"running", "done", "exhausted", "failed", "stopped"})
CELL_STATUSES = frozenset({"running", "done", "failed", "stopped"})
# The fields of each block, and of each step and sub-agent drive it copies, that the page
# draws: each of one of its types, or one of its texts. A Library change's fields, and a
# worked example's left out, show as text, whatever they hold.
BLOCKS = {
    "text": {"text": TEXT},
    "step": {
        "turn": (int,),
        "node": NODE,
        "index": (int,),
        "task": MAYBE_TEXT,
        "note": MAYBE_TEXT,
        "step": (dict,),
    },
    "agent": {"turn": (int,), "node": NODE, "note": MAYBE_TEXT, "agents": (list,)},
    "table": {"columns": (list,), "rows": (list,)},
}
STEPS = {
    "think": {"text": TEXT},
    "cell": {
        "code": TEXT,
        "output": MAYBE_TEXT,
        "status": CELL_STATUSES,
        "agents": (list,),
    },
    "change": {},
    "page": {"page": TEXT, "title": TEXT},
    "left_out": {},
}
DRIVE = {
    "node": NODE,
    "drive": MAYBE_NUMBER,
    "task": TEXT,
    "namespace": MAYBE_TEXT,
    "model": MAYBE_TEXT,
    "status": DRIVE_STATUSES,
    "result": MAYBE_TEXT,
    "seconds": MAYBE_NUMBER,
    "cost": MAYBE_NUMBER,
    "unpriced": (bool,),
    "steps": (list,),
}


def shape_of(value: Any, shapes: dict[str, dict], key: str) -> dict | None:
    """The shape in *shapes* that *value*'s *key* names, if it is a dict naming one."""
    name = value.get(key) if type(value) is dict else None
    return shapes.get(name) if type(name) is str else None


def fits(value: dict, shape: dict[str, tuple | frozenset]) -> bool:
    """Whether each field of *shape* in *value* is of one of its types, exactly, so a bool
    is no number; or one of its texts."""
    return all(allows(allowed, value.get(key)) for key, allowed in shape.items())


def allows(allowed: tuple | frozenset, field: Any) -> bool:
    if isinstance(allowed, frozenset):
        return type(field) is str and field in allowed
    return type(field) in allowed


def table_fits(columns: list, rows: list) -> bool:
    return (
        len(columns) <= MAX_COLUMNS
        and len(rows) <= MAX_ROWS
        and all(type(column) is str for column in columns)
        and all(
            type(row) is list
            and len(row) <= MAX_COLUMNS
            and all(type(value) is str for value in row)
            for row in rows
        )
    )


def record_fits(steps: list, drives: list) -> bool:
    """Whether *steps* and sub-agent *drives* copied from a record, and all below them,
    are shaped as fold.js folds them; walked, not recursed, as a file may nest them deep."""
    stack = [(STEPS, step) for step in steps] + [(DRIVE, drive) for drive in drives]
    while stack:
        shapes, value = stack.pop()
        if shapes is DRIVE:
            if type(value) is not dict or not fits(value, DRIVE):
                return False
            stack += [(STEPS, step) for step in value["steps"]]
            continue
        shape = shape_of(value, STEPS, "kind")
        if shape is None or not fits(value, shape):
            return False
        if value["kind"] == "cell":
            stack += [(DRIVE, drive) for drive in value["agents"]]
    return True


def well_formed(block: Any) -> bool:
    """Whether *block* is one the page tool could have saved: of a known type, each field
    of its type and within its limits."""
    shape = shape_of(block, BLOCKS, "type")
    if shape is None or not fits(block, shape):
        return False
    texts = [block.get(key) or "" for key in ("text", "note") if key in shape]
    if any(len(text) > MAX_TEXT_CHARS for text in texts):
        return False
    if block["type"] == "table":
        return table_fits(block["columns"], block["rows"])
    if block["type"] == "step":
        return record_fits([block["step"]], [])
    return block["type"] == "text" or record_fits([], block["agents"])


def stored_text(blocks: Any) -> str | None:
    """*blocks* as the pages table holds them, if the reader reads them back as they are:
    None for a value JSON doesn't keep, a number past ±2**53, a block the page tool could
    not have saved, or nesting too deep to write or read."""
    try:
        text = json.dumps(blocks)
        read = loaded_blocks(text)
        same = read == blocks and all(map(well_formed, read))
    except UNREADABLE:
        return None
    return text if same else None


def add_page(
    connection: sqlite3.Connection,
    chat: str,
    run: str,
    node: int | str | None,
    title: str,
    blocks: str,
) -> str:
    """Save a page of *chat*'s, which agent *node* made in *run*, holding *blocks* as
    JSON, under a readable id no other page has; its id."""
    page = {
        "id": generate_slug(ID_WORDS),
        "chat": chat,
        "run": run,
        "node": node,
        "title": title,
        "blocks": blocks,
        "created": time.time(),
    }
    with writing(connection):
        for _ in range(ID_TRIES):
            if inserted(connection, page):
                return page["id"]
            page["id"] = generate_slug(ID_WORDS)
    raise LibraryError(NO_FREE_ID)


def update_page(
    connection: sqlite3.Connection,
    chat: str,
    run: str,
    page: str,
    blocks: str,
) -> None:
    """Save *blocks*, as JSON, in place of those of page *page*, which *chat*'s run *run*
    made. A refusal, as a shared file's own constraints may make, is a LibraryError."""
    try:
        with writing(connection):
            updated = connection.execute(
                "UPDATE pages SET blocks = ? WHERE id = ? AND chat = ? AND run = ?",
                (blocks, page, chat, run),
            )
    except sqlite3.IntegrityError as error:
        raise refused(error) from error
    if updated.rowcount == 0:
        raise LibraryError(PAGE_GONE)


def inserted(connection: sqlite3.Connection, page: dict[str, Any]) -> bool:
    """Whether *page* went in: False when its id is taken. Any other refusal, as a shared
    file's own constraints may make, is a LibraryError."""
    try:
        connection.execute(ADD_PAGE, page)
    except sqlite3.IntegrityError as error:
        if error.sqlite_errorname == ID_TAKEN:
            return False
        raise refused(error) from error
    return True


def refused(error: sqlite3.IntegrityError) -> LibraryError:
    return LibraryError(f"The Library file refused the page: {error}.")


def page_exists(connection: sqlite3.Connection, page: str) -> bool:
    if not re.fullmatch(PAGE_ID, page):
        return False
    found = connection.execute("SELECT 1 FROM pages WHERE id = ?", (page,))
    return found.fetchone() is not None


def loaded_blocks(text: Any) -> list | None:
    """The blocks a page's row holds as JSON, or None when the page tool could not have
    saved them."""
    if type(text) is not str or len(text) > MAX_PAGE_CHARS:
        return None
    try:
        blocks = json.loads(text, **NUMBERS)
    except UNREADABLE:
        return None
    return blocks if type(blocks) is list and len(blocks) <= MAX_BLOCKS else None


def read_page(connection: sqlite3.Connection, page: str) -> dict[str, Any] | None:
    """Page *page* as its tab draws it: its title, and each block the page tool could have
    saved. None when there is no such page, or its row is one the tool could not have
    saved."""
    if not re.fullmatch(PAGE_ID, page):
        return None
    row = connection.execute(
        "SELECT title, blocks FROM pages WHERE id = ?", (page,)
    ).fetchone()
    if row is None:
        return None
    title, blocks = row[0], loaded_blocks(row[1])
    if type(title) is not str or len(title) > MAX_TITLE_CHARS or blocks is None:
        log.warning("pages.left_out", page=page)
        return None
    kept = []
    for number, block in enumerate(blocks):
        if well_formed(block):
            kept.append(block)
        else:
            log.warning("pages.block_left_out", page=page, block=number)
    return {"id": page, "title": title, "blocks": kept}


def page_text(connection: sqlite3.Connection, page: str) -> str | None:
    """Page *page* as read_page gives it, written as JSON for its tab; None when there is
    none, or it nests deeper than JSON writes, as a shared file's may."""
    found = read_page(connection, page)
    if found is None:
        return None
    try:
        return json.dumps(found)
    except UNREADABLE:
        log.warning("pages.left_out", page=page)
        return None
