"""The agent's own tools: the app gives every run these, besides the Library's.

The engine builds each from this file by its path (factory_from), in the worker, with only
what its block in the run's config holds: the app file's path, the chat, the run, the chat's
namespace and the variables that hold keys. Each opens the app file as it is, never making
a moved or deleted one afresh.

history reads the chat's earlier runs from the app file, read-only, as the page reads them:
runs.py checks each row, and folded() folds a run's events into its tree as fold.js does.

page builds a page for the user from the same trees: each step or sub-agent it points to
is copied from the record, as fold.js folds it, into the pages table; a page opened is
logged as page.open, which the worker records as a run event.

library reads and writes the Library through library.py, with the panel's checks and
transactions; the locks are here, as the panel may still edit root, a namespace's repl and
reasoner, and which namespaces grant the app's own tools. It writes only in a run whose
chat's namespace grants it, so a sub-agent spawned into such a namespace from another chat
can't, and never a key. It guards the tool, not the file: a cell can open the file itself.
A tool it writes is built first, as a run builds it, from a file in the run's tools folder.
Each change is logged as library.change, which the worker records as a run event.
"""

import ast
import copy
import json
import keyword
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import structlog
from deep_reasoner.namespaces import ROOT, NamespaceConfig, NamespaceRegistry
from deep_reasoner.primitives import RESERVED_NAMES, Func
from deep_reasoner.tools import ToolBuildError
from deep_reasoner.v2.cli import V2Config, make_tool

from deep_reasoning.library import (
    APP_TOOLS,
    BUILDERS,
    HISTORY_DESCRIPTION,
    LIBRARY_DESCRIPTION,
    LIBRARY_TOOL,
    PAGE_DESCRIPTION,
    LibraryError,
    add_decomposition,
    add_namespace,
    change_namespace,
    config_and_registry,
    count,
    distrust,
    dump,
    loaded,
    namespace_rows,
    parsed_python,
    reading,
    store_tool,
    tool_file,
    tool_text,
)
from deep_reasoning.pages import (
    DRIVE,
    MAX_BLOCKS,
    MAX_COLUMNS,
    MAX_PAGE_CHARS,
    MAX_ROWS,
    MAX_TEXT_CHARS,
    MAX_TITLE_CHARS,
    NODE,
    add_page,
    page_link,
    stored_text,
)
from deep_reasoning.runs import (
    CHANGE_FIELDS,
    PAGE_FIELDS,
    chat_turns,
    secrets_of,
    total_cost,
    withheld,
)

log = structlog.get_logger()

Event = dict[str, Any]
Holder = dict[str, Any]
STARTED_WITH = ("node", "parent", "drive", "task", "namespace")
AGENT_FIELDS = (*STARTED_WITH, "status", "result")
ENDED_WITH = ("outcome", "answer", "detail", "cost", "seconds")
LOOKUP_FIELDS = ("node", "parent", "cell")
ROOT_LOCKED = (
    "root is locked: the library tool never writes it. Write into another namespace."
)
NOT_THIS_CHAT = (
    "The Library can be changed only in a chat whose namespace grants the library tool;"
    " pick librarian under the message box."
)
NOT_TEXT = "The namespace's name and the YAML should be text."
# As the panel's limit on a decomposition's file; past 64 MiB, its change event would end
# the run after the write.
MAX_YAML_CHARS = 1_000_000
TOO_LONG = f"The YAML may be at most {MAX_YAML_CHARS:,} characters."
HOLDS_A_KEY = (
    "What you wrote holds a provider's key, and the Library file is shared, so keys never"
    " go in it. Take the key out."
)
# reasoner: claude_code would bill the user's subscription.
LOCKED_FIELDS = ("repl", "reasoner")
USERS_CHOICE = "the user's choice, in the Library panel."
# A tool of this factory, or a reasoner of this type, runs on the user's subscription.
CLAUDE_CODE = "claude_code"
NOT_A_TOOL = (
    "name, source and factory should be text, and settings a dict of plain values (text,"
    " numbers, True, False, None, and lists and dicts of them keyed by text, each one"
    " used once), or None."
)
PLAIN = (str, int, float, bool, type(None))
# Deeper settings are refused, as YAML writes them by recursion.
MAX_NESTING = 20
# A run hands the factory every key of the tool's block but these.
OWN_KEYS = ("factory", "factory_from")
SETS_OWN_KEYS = "settings may not hold factory or factory_from: write_tool sets them."
TAKEN_NAMES = (*BUILDERS, *APP_TOOLS, *sorted(RESERVED_NAMES))
TOOL_TOO_LONG = (
    f"The source, and the settings as YAML, may each be at most {MAX_YAML_CHARS:,}"
    " characters."
)
CHECKS = "checks"
# Any other REPL is a sandbox, and building a tool runs its code in the worker, outside it.
LOCAL_REPL = "local"
NOT_LOCAL = (
    "write_tool works only in a chat whose namespace runs the local REPL: building the"
    " tool to check it runs its code outside any sandbox."
)
TITLE = f"A page's title should be text of 1 to {MAX_TITLE_CHARS} characters."
NOT_MARKDOWN = "text takes Markdown, as text (a str)."
NOT_A_REF = (
    "node should be a sub-agent's node as history.agents(turn) gives it (None for the"
    " top agent), and note Markdown text or None."
)
NOT_AN_INDEX = "index should be a step's number in history.steps(turn, node), from 0."
NOT_ROWS = (
    "rows should be a list of dicts keyed by text, or a list of lists; columns a list"
    " of texts, or None."
)
NOT_VALUES = "Each value in a table should be text, a number, True, False or None."
NOT_SHOWN = (
    "That can't be shown: the record it copies isn't one a run of the app writes."
)
NOT_SAVED = (
    "The page can't be saved as it is: open() saves only what the page's methods added."
)
TOO_DEEP = "That nests too deep to show on a page."
EMPTY = "The page is empty: add text, a step, an agent or a table first."


@dataclass
class Tree:
    """A run's events folded: the root agent's drive, each sub-agent drive as a row, and
    each node's current drive (its holder) with its parent's node."""

    root: Holder = field(
        default_factory=lambda: {"node": None, "drive": None, "costs": [], "steps": []}
    )
    rows: list[Holder] = field(default_factory=list)
    nodes: dict[Any, tuple[Holder, Any]] = field(default_factory=dict)
    cells: dict[Any, dict[str, Any]] = field(default_factory=dict)
    # The drives each cell started, by the cell's id(), for a page's copy of it.
    started: dict[int, list[Holder]] = field(default_factory=dict)


def start_agent(tree: Tree, event: Event) -> None:
    node, parent = event.get("node"), event.get("parent")
    if parent is None:
        tree.root |= {"node": node, "drive": event.get("drive")}
        tree.nodes[node] = (tree.root, None)
        return
    cell = tree.cells.get(event.get("cell"))
    if cell is None:
        return
    row = {key: event.get(key) for key in STARTED_WITH}
    row |= {"status": "running", "result": None, "costs": [], "steps": []}
    # When it started, for a page, which times it as fold.js does; the cell it started
    # in, so the cell's id() in tree.started names no other while the row lives.
    row |= {"started": event["t"], "seconds": None, "cell": cell}
    if node not in cell["agents"]:
        cell["agents"].append(node)
    tree.rows.append(row)
    tree.started.setdefault(id(cell), []).append(row)
    tree.nodes[node] = (row, parent)


def holder_of(tree: Tree, node: Any) -> Holder | None:
    return tree.nodes.get(node, (None, None))[0]


def add_step(tree: Tree, node: Any, step: dict[str, Any]) -> None:
    holder = holder_of(tree, node)
    if holder is not None:
        step["drive"] = holder["drive"]
        holder["steps"].append(step)


def think(tree: Tree, event: Event) -> None:
    add_step(tree, event.get("node"), {"kind": "think", "text": event.get("text")})


def start_cell(tree: Tree, event: Event) -> None:
    cell = {"kind": "cell", "code": event.get("code"), "output": None}
    cell |= {"status": "running", "agents": []}
    tree.cells[event.get("cell")] = cell
    add_step(tree, event.get("node"), cell)


def add_logged(kind: str, fields: tuple[str, ...], tree: Tree, event: Event) -> None:
    """A change the library tool made, or a page the page tool opened, as a step of the
    agent that did it."""
    step = {"kind": kind} | {key: event.get(key) for key in fields}
    add_step(tree, event.get("node"), step)


def end_cell(tree: Tree, event: Event) -> None:
    cell = tree.cells.get(event.get("cell"))
    if cell is not None:
        cell |= {key: event.get(key) for key in ("code", "output", "status")}


def add_usage(tree: Tree, event: Event) -> None:
    """A call's cost counts for its agent and every agent above it."""
    node, seen = event.get("node"), set()
    # Each once: a file from someone else may name a node its own ancestor.
    while node in tree.nodes and node not in seen:
        seen.add(node)
        holder, node = tree.nodes[node]
        holder["costs"].append(event.get("cost"))


def end_agent(tree: Tree, event: Event) -> None:
    row = holder_of(tree, event.get("node"))
    if row is None or row is tree.root:
        return
    result = event.get("result")
    row["status"] = event.get("status")
    row["result"] = event.get("detail") if result is None else result
    row["seconds"] = event["t"] - row["started"]


def end_turn(tree: Tree, event: Event) -> None:
    """What the run left going when it ended was stopped with it, a sub-agent timed to
    the end."""
    for row in tree.rows:
        if row["status"] == "running":
            row |= {"status": "stopped", "seconds": event["t"] - row["started"]}
    for cell in tree.cells.values():
        if cell["status"] == "running":
            cell["status"] = "stopped"


FOLDS = {
    "agent.start": start_agent,
    "think": think,
    "cell.start": start_cell,
    "cell.end": end_cell,
    "library.change": partial(add_logged, "change", CHANGE_FIELDS),
    "page.open": partial(add_logged, "page", PAGE_FIELDS),
    "usage": add_usage,
    "agent.end": end_agent,
    "turn.end": end_turn,
}


def well_keyed(event: Event) -> bool:
    """Whether *event*'s node, parent and cell are ones a run of the app's could have."""
    return all(isinstance(event.get(key), int | str | None) for key in LOOKUP_FIELDS)


def folded(events: list[Event]) -> Tree:
    """A run's events as the tree fold.js folds them into for the page. An event that
    isn't well keyed, as a file from someone else may hold, is left out."""
    tree = Tree()
    for event in events:
        fold = FOLDS.get(event["kind"])
        if fold is None:
            continue
        if not well_keyed(event):
            log.warning("history.event_left_out", seq=event["seq"])
            continue
        fold(tree, event)
    return tree


def turn_row(number: int, events: list[Event], tree: Tree) -> dict[str, Any]:
    start, end = events[0], events[-1]
    return {
        "turn": number,
        "message": start["prompt"],
        "namespace": start.get("namespace"),
        "model": start.get("model"),
        **{key: end.get(key) for key in ENDED_WITH},
        "agents": len(tree.rows),
    }


def agent_row(row: Holder) -> dict[str, Any]:
    fields = {key: row[key] for key in AGENT_FIELDS}
    return fields | {"cost": total_cost(row["costs"]), "steps": len(row["steps"])}


def open_existing(library: str | Path, mode: str = "ro") -> sqlite3.Connection:
    """The app file at *library* as it is, to read ("ro") or to write ("rw"): unlike
    open_library, it never makes a moved or deleted file afresh. It runs none of the file's
    own triggers or views, and wipes what it deletes. In WAL mode a reader holds up no event
    the server saves meanwhile."""
    connection = sqlite3.connect(
        f"{Path(library).as_uri()}?mode={mode}", uri=True, isolation_level=None
    )
    connection.execute("PRAGMA secure_delete = ON")
    distrust(connection)
    return connection


@contextmanager
def app_file(library: str, mode: str) -> Iterator[sqlite3.Connection]:
    """The app file as it is, for one use; a SQLite failure (busy, read only, moved or
    deleted) as a LibraryError in words the model can pass on."""
    try:
        with closing(open_existing(library, mode)) as connection:
            yield connection
    except sqlite3.OperationalError as error:
        raise LibraryError(
            f"The Library file can't be used now ({error}); try again."
        ) from error


def earlier_runs(library: str, chat: str, run: str) -> list[list[Event]]:
    """The events of each of *chat*'s runs but *run*, oldest first, as the page reads
    them: a run that didn't start is left out, and one with no turn.end ends failed."""
    with closing(open_existing(library)) as connection:
        turns = chat_turns(connection, chat)
    return [turn["events"] for turn in turns if turn["run"] != run]


class History:
    """The chat's earlier runs, as run *run* of *chat* in the app file *library* sees
    them: read once, folded, and numbered from 1. The history tool's methods, and the
    trees the page tool copies from."""

    def __init__(self, library: str, chat: str, run: str) -> None:
        self._where = (library, chat, run)

    @cached_property
    def _runs(self) -> list[tuple[list[Event], Tree]]:
        # Read once: no run of the chat's starts or ends while this one is going.
        return [(events, folded(events)) for events in earlier_runs(*self._where)]

    def tree(self, turn: int) -> Tree:
        """Message *turn*'s run, folded."""
        runs = self._runs
        if not runs:
            raise ValueError("This chat has no earlier messages: this is its first.")
        if type(turn) is not int or not 1 <= turn <= len(runs):
            raise ValueError(
                f"There is no message {turn!r}: this chat's earlier messages are"
                f" numbered 1 to {len(runs)}."
            )
        return runs[turn - 1][1]

    def drives(self, turn: int, node: Any) -> list[Holder]:
        """Message *turn*'s top agent, or each drive of its sub-agent *node*."""
        tree = self.tree(turn)
        if node is None or node == tree.root["node"]:
            return [tree.root]
        drives = [row for row in tree.rows if row["node"] == node]
        if not drives:
            nodes = [
                tree.root["node"],
                *dict.fromkeys(row["node"] for row in tree.rows),
            ]
            raise ValueError(
                f"Message {turn} has no agent {node!r}: its agents' nodes are {nodes}."
            )
        return drives

    def turns(self) -> list[dict[str, Any]]:
        """This chat's earlier messages, numbered from 1, with how each ended."""
        return [turn_row(number, *each) for number, each in enumerate(self._runs, 1)]

    def agents(self, turn: int) -> list[dict[str, Any]]:
        """Each sub-agent drive of message *turn*."""
        return [agent_row(row) for row in self.tree(turn).rows]

    def steps(self, turn: int, node: Any = None) -> list[dict[str, Any]]:
        """The thoughts, cells, Library changes and pages of message *turn*'s top agent,
        or of sub-agent *node*."""
        # A copy: the folded runs are kept for the next call.
        drives = self.drives(turn, node)
        return copy.deepcopy([step for row in drives for step in row["steps"]])


def build_history(client: Any, params: dict[str, Any]) -> Func:
    """The history tool of run *run* of *chat*, in the app file *library*, as its block
    in the run's config names them."""
    history = History(params["library"], params["chat"], params["run"])
    methods = SimpleNamespace(
        turns=history.turns, agents=history.agents, steps=history.steps
    )
    return Func(methods, description=HISTORY_DESCRIPTION)


def shown_drive(tree: Tree, row: Holder) -> dict[str, Any]:
    """A sub-agent's drive as fold.js folds it, for a page: its steps, and its own
    sub-agents' drives under the cells that started them."""
    cost = total_cost(row["costs"])
    return {key: row.get(key) for key in DRIVE} | {
        "cost": cost if row["costs"] else None,
        "unpriced": cost is None,
        "steps": [shown_step(tree, step) for step in row["steps"]],
    }


def shown_step(tree: Tree, step: dict[str, Any]) -> dict[str, Any]:
    """A step as fold.js folds it, for a page: a cell's sub-agents as their drives."""
    if step["kind"] != "cell":
        return step
    drives = tree.started.get(id(step), [])
    return step | {"agents": [shown_drive(tree, row) for row in drives]}


def copied(show: Callable[[Tree, Any], Any], tree: Tree, item: Any) -> Any:
    """*item* as *show* copies it from *tree*; a ValueError when it nests too deep to
    copy, as a record in a shared file may."""
    try:
        return show(tree, item)
    except RecursionError as error:
        raise ValueError(TOO_DEEP) from error


def keys_withheld(value: Any, secrets: list[str]) -> Any:
    """*value* with each of *secrets* withheld; a ValueError when it nests too deep to
    walk."""
    try:
        return withheld(value, secrets)
    except RecursionError as error:
        raise ValueError(TOO_DEEP) from error


def checked_title(title: Any, secrets: list[str]) -> str:
    title = withheld(title, secrets) if type(title) is str else ""
    if not title.strip() or len(title) > MAX_TITLE_CHARS:
        raise ValueError(TITLE)
    return title


def table_block(rows: Any, columns: Any) -> dict[str, Any]:
    """*rows*, a list of dicts or of lists, as a table block under *columns* (by default
    the dicts' keys, or none), each value as text."""
    shaped = type(rows) is list and (
        all(type(row) is dict for row in rows) or all(type(row) is list for row in rows)
    )
    named = columns is None or (
        type(columns) is list and all(type(each) is str for each in columns)
    )
    if not shaped or not named:
        raise ValueError(NOT_ROWS)
    if len(rows) > MAX_ROWS:
        raise ValueError(
            f"A table has at most {MAX_ROWS:,} rows; this has {len(rows):,}."
        )
    if rows and type(rows[0]) is dict:
        if any(type(key) is not str for row in rows for key in row):
            raise ValueError(NOT_ROWS)
        if columns is None:
            columns = list(dict.fromkeys(key for row in rows for key in row))
        rows = [[row.get(column) for column in columns] for row in rows]
    width = max(map(len, [columns or [], *rows]))
    if width > MAX_COLUMNS:
        raise ValueError(
            f"A table has at most {MAX_COLUMNS} columns; this has {width}."
        )
    if any(type(value) not in PLAIN for row in rows for value in row):
        raise ValueError(NOT_VALUES)
    texts = [["" if value is None else str(value) for value in row] for row in rows]
    return {"type": "table", "columns": columns or [], "rows": texts}


class Page:
    """A page in progress, as page.new makes it: each method adds a block, and open()
    saves it. What it holds has the worker's keys withheld, as run events do."""

    def __init__(
        self,
        title: str,
        history: History,
        secrets: list[str],
        save: Callable[[str, list[dict[str, Any]]], str],
    ) -> None:
        self._title = checked_title(title, secrets)
        self._blocks = []
        self._history, self._secrets, self._save = history, secrets, save
        self._chars = len(json.dumps([]))

    def text(self, markdown: str) -> None:
        """Add *markdown*, a paragraph or more."""
        if type(markdown) is not str:
            raise ValueError(NOT_MARKDOWN)
        self._add({"type": "text", "text": markdown})

    def step(
        self, turn: int, node: Any = None, *, index: int, note: str | None = None
    ) -> None:
        """Add history.steps(turn, node)[index], drawn as the chat draws it, with a
        Markdown *note*."""
        check_ref(node, note)
        if type(index) is not int:
            raise ValueError(NOT_AN_INDEX)
        drives = self._history.drives(turn, node)
        steps = [(row, step) for row in drives for step in row["steps"]]
        if not -len(steps) <= index < len(steps):
            who = "top agent" if node is None else f"agent {node!r}"
            raise ValueError(
                f"Message {turn}'s {who} has {count(len(steps), 'step')}, numbered from"
                f" 0: there is no step {index}."
            )
        row, step = steps[index]
        block = {
            "type": "step",
            "turn": turn,
            "node": node,
            "index": index % len(steps),
            "task": row.get("task"),
            "note": note,
            "step": copied(shown_step, self._history.tree(turn), step),
        }
        self._add(block)

    def agent(self, turn: int, node: Any, note: str | None = None) -> None:
        """Add message *turn*'s sub-agent *node*, each drive of it with its steps and its
        own sub-agents' below, with a Markdown *note*."""
        check_ref(node, note)
        tree = self._history.tree(turn)
        drives = self._history.drives(turn, node)
        if drives[0] is tree.root:
            raise ValueError(
                f"Message {turn}'s top agent is no sub-agent: show its steps with"
                f" p.step({turn}, index=...)."
            )
        block = {
            "type": "agent",
            "turn": turn,
            "node": node,
            "note": note,
            "agents": [copied(shown_drive, tree, row) for row in drives],
        }
        self._add(block)

    def table(self, rows: list, columns: list[str] | None = None) -> None:
        """Add *rows*, a list of dicts or of lists, as a table under *columns*: by default
        the dicts' keys, and none for lists."""
        self._add(table_block(rows, columns))

    def open(self) -> str:
        """Save the page; its link, which the chat shows under the answer. What it saves
        is checked again, as a cell may have set the page's state itself."""
        title = checked_title(self._title, self._secrets)
        if not self._blocks:
            raise ValueError(EMPTY)
        blocks = keys_withheld(self._blocks, self._secrets)
        if stored_text(blocks) is None:
            raise ValueError(NOT_SAVED)
        page = self._save(title, blocks)
        log.info("page.open", page=page, title=title)
        return page_link(page)

    def _add(self, block: dict[str, Any]) -> None:
        if len(self._blocks) == MAX_BLOCKS:
            raise ValueError(
                f"A page holds at most {MAX_BLOCKS} blocks: open this one, and put the"
                " rest on another."
            )
        block = keys_withheld(block, self._secrets)
        for key in ("text", "note"):
            if len(block.get(key) or "") > MAX_TEXT_CHARS:
                raise ValueError(
                    f"A {key} may be at most {MAX_TEXT_CHARS:,} characters; this one"
                    f" is {len(block[key]):,}."
                )
        # As it reads back once saved: a number a crafted record sums past ±2**53 won't.
        stored = stored_text([block])
        if stored is None:
            raise ValueError(NOT_SHOWN)
        chars = len(stored) - len("[]") + (len(", ") if self._blocks else 0)
        if self._chars + chars > MAX_PAGE_CHARS:
            raise ValueError(
                f"With this, the page would be {self._chars + chars:,} characters as"
                f" saved, past {MAX_PAGE_CHARS:,}: show less, or put it on another page."
            )
        self._blocks.append(block)
        self._chars += chars


def check_ref(node: Any, note: Any) -> None:
    if type(node) not in NODE or not (note is None or type(note) is str):
        raise ValueError(NOT_A_REF)


def build_page(client: Any, params: dict[str, Any]) -> Func:
    """The page tool of run *run* of *chat*, in the app file *library*, whose worker holds
    keys in *key_names*, as its block in the run's config names them."""
    library, chat, run = params["library"], params["chat"], params["run"]
    history = History(library, chat, run)
    secrets = secrets_of(os.environ, params["key_names"])

    def save(title: str, blocks: list[dict[str, Any]]) -> str:
        # The agent whose cell opens the page, as its page.open event names it.
        node = structlog.contextvars.get_contextvars().get("agent_id")
        with app_file(library, "rw") as connection:
            return add_page(connection, chat, run, node, title, blocks)

    def new(title: str) -> Page:
        """A page titled *title*, saved when its open() is called."""
        return Page(title, history, secrets, save)

    return Func(SimpleNamespace(new=new), description=PAGE_DESCRIPTION)


def texts_in(data: Any) -> Iterator[str]:
    """Every text in parsed YAML *data*, keys included; each mapping or list once, as
    aliases may share or nest them."""
    stack, seen = [data], set()
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict | list) and id(value) not in seen:
            seen.add(id(value))
            stack.extend(
                [*value, *value.values()] if isinstance(value, dict) else value
            )


def check_no_key(texts: list[str], key_names: list[str]) -> None:
    """Raise if any of *texts* holds the value of a key variable in *key_names*."""
    secrets = secrets_of(os.environ, key_names)
    if any(secret in text for secret in secrets for text in texts):
        raise LibraryError(HOLDS_A_KEY)


def check_writable(namespace: str, yaml: str, key_names: list[str]) -> None:
    """Raise unless the tool may write *yaml* for the namespace named *namespace*."""
    # Exactly str: a subclass could compare unequal to "root" and still name it.
    if type(namespace) is not str or type(yaml) is not str:
        raise LibraryError(NOT_TEXT)
    if namespace == ROOT:
        raise LibraryError(ROOT_LOCKED)
    if len(yaml) > MAX_YAML_CHARS:
        raise LibraryError(TOO_LONG)
    # Parsed too, as an escape such as "\x73k-" spells a key the raw text doesn't hold.
    check_no_key([namespace, yaml, *texts_in(loaded(yaml))], key_names)


def plain(value: Any) -> bool:
    """Whether *value* is text, a number, True, False or None, or a list or dict of them
    keyed by text, nested at most MAX_NESTING deep, each list and dict in it once (so
    none holds itself)."""
    stack, seen = [(value, 0)], set()
    while stack:
        each, depth = stack.pop()
        if type(each) in PLAIN:
            continue
        if type(each) not in (list, dict) or id(each) in seen or depth == MAX_NESTING:
            return False
        seen.add(id(each))
        if type(each) is dict and any(type(key) is not str for key in each):
            return False
        items = each.values() if type(each) is dict else each
        stack.extend((item, depth + 1) for item in items)
    return True


def check_tool(
    name: str, source: str, factory: str, settings: Any, key_names: list[str]
) -> None:
    """Raise unless the tool may write tool *name*, built by the function *factory* in
    the Python *source* with *settings*, before it is built."""
    texts = [name, source, factory]
    is_plain = settings is None or type(settings) is dict and plain(settings)
    if any(type(each) is not str for each in texts) or not is_plain:
        raise LibraryError(NOT_A_TOOL)
    if settings and any(key in settings for key in OWN_KEYS):
        raise LibraryError(SETS_OWN_KEYS)
    tool_file(name)
    if keyword.iskeyword(name) or name in TAKEN_NAMES:
        raise LibraryError(
            f'"{name}" is taken: it is a Python keyword, a built-in or app tool, or a name'
            f" the REPL binds itself ({', '.join(TAKEN_NAMES)}). Pick another."
        )
    if max(len(source), len(dump(settings))) > MAX_YAML_CHARS:
        raise LibraryError(TOOL_TOO_LONG)
    module = parsed_python(source)
    # Its literals too, as an escape can spell a key the raw text doesn't hold.
    literals = [
        node.value if type(node.value) is str else node.value.decode("latin-1")
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and type(node.value) in (str, bytes)
    ]
    check_no_key([*texts, *texts_in(settings), *literals], key_names)


def check_tool_locks(
    config: V2Config, registry: NamespaceRegistry, chat_namespace: str, name: str
) -> None:
    """Raise if the tool may not write tool *name* in a chat in *chat_namespace*: the
    chat's REPL is a sandbox, root grants the tool, or the engine builds it from its
    block alone, which the tool couldn't write back."""
    if registry.resolve(chat_namespace).repl.type != LOCAL_REPL:
        raise LibraryError(NOT_LOCAL)
    if name in registry.config(ROOT).tools:
        raise LibraryError(f"root grants {name}: replacing it is {USERS_CHOICE}")
    block = config.tools.get(name)
    if block is not None and "factory_from" not in block:
        raise LibraryError(
            f"{name} is built by the engine's own {block.get('factory', name)} factory,"
            f" which write_tool can't write back: replacing it is {USERS_CHOICE}"
        )


def claude_code_tools(config: V2Config, tools: Iterable[str]) -> list[str]:
    """Those of *tools* that run Claude Code, sorted."""
    return sorted(
        tool
        for tool in tools
        if config.tools.get(tool, {}).get("factory", tool) == CLAUDE_CODE
    )


def check_locks(
    config: V2Config, registry: NamespaceRegistry, namespace: NamespaceConfig
) -> None:
    """Raise if *namespace* sets or changes what only the user may: its repl or reasoner,
    which of the app's own tools it grants, or a tool that runs Claude Code, granted
    afresh; or, when new, would inherit library or Claude Code."""
    name = namespace.name
    new = name not in registry.names
    stored = NamespaceConfig(name=name) if new else registry.config(name)
    for key in LOCKED_FIELDS:
        if getattr(namespace, key) != getattr(stored, key):
            raise LibraryError(
                f"{key} may not be set or changed: a namespace's repl and reasoner are"
                f" {USERS_CHOICE}"
            )
    changed = sorted((set(namespace.tools) ^ set(stored.tools)) & APP_TOOLS.keys())
    if changed:
        raise LibraryError(
            f"tools may not add or take out {' and '.join(changed)}: which namespaces"
            f" grant the app's own tools is {USERS_CHOICE}"
        )
    claude = claude_code_tools(config, set(namespace.tools) - set(stored.tools))
    if claude:
        raise LibraryError(
            f"tools may not add {' and '.join(claude)}, which runs Claude Code on the"
            f" user's subscription: granting it is {USERS_CHOICE}"
        )
    above = name.rpartition(".")[0] or ROOT
    # A change keeps its chain, and a missing parent is refused after this.
    if new and above in registry.names:
        check_inherited(config, registry, name, above)


def check_inherited(
    config: V2Config, registry: NamespaceRegistry, name: str, above: str
) -> None:
    """Raise if new namespace *name*, under *above*, would inherit library or Claude
    Code, as its backbone or a tool."""
    resolved = registry.resolve(above)
    if LIBRARY_TOOL in resolved.tools:
        raise LibraryError(
            f"{name} would inherit library from {resolved.sources.tools[LIBRARY_TOOL]}:"
            f" which namespaces grant the app's own tools is {USERS_CHOICE}"
        )
    if registry.backbone(above) == CLAUDE_CODE:
        level = resolved.sources.reasoner or "the run settings"
        raise LibraryError(
            f"{name} would inherit the claude_code reasoner from {level}, which runs on"
            f" the user's subscription: which namespaces run Claude Code is {USERS_CHOICE}"
        )
    claude = claude_code_tools(config, resolved.tools)
    if claude:
        tool = claude[0]
        raise LibraryError(
            f"{name} would inherit {tool}, which runs Claude Code on the user's"
            f" subscription, from {resolved.sources.tools[tool]}: which namespaces have it"
            f" is {USERS_CHOICE}"
        )


def logged(done: str, before: str | None, after: str, **change: str) -> Event:
    """Log a change the tool made as library.change; what it did, for the model."""
    log.info("library.change", **change, before=before, after=after)
    return {"done": done, **change}


def build_library(client: Any, params: dict[str, Any]) -> Func:
    """The library tool, over the app file *library* its block in the run's config names,
    for a run of a chat in *namespace*, whose worker holds keys in *key_names*; it builds
    a tool it writes with the run's *client*, in the run's *tools_dir*."""
    library, chat_namespace = params["library"], params["namespace"]
    key_names, tools_dir = params["key_names"], Path(params["tools_dir"])

    def show() -> list[dict[str, Any]]:
        """Every namespace, as the Library panel shows it."""
        with app_file(library, "ro") as connection, reading(connection):
            return namespace_rows(connection, *config_and_registry(connection))

    def check_grant(registry: NamespaceRegistry) -> None:
        """Raise unless this run's chat is in a namespace that grants the library tool, as
        the Library stands under the write lock: a grant taken away meanwhile counts."""
        granted = chat_namespace in registry.names and (
            LIBRARY_TOOL in registry.resolve(chat_namespace).tools
        )
        if not granted:
            raise LibraryError(NOT_THIS_CHAT)

    def may_write(connection: sqlite3.Connection) -> None:
        check_grant(config_and_registry(connection)[1])

    def may_change(connection: sqlite3.Connection, namespace: NamespaceConfig) -> None:
        config, registry = config_and_registry(connection)
        check_grant(registry)
        check_locks(config, registry, namespace)

    def may_write_tool(connection: sqlite3.Connection, name: str) -> None:
        config, registry = config_and_registry(connection)
        check_grant(registry)
        check_tool_locks(config, registry, chat_namespace, name)

    def write_decomposition(namespace: str, yaml: str) -> dict[str, Any]:
        """Add the decomposition *yaml* to *namespace*, or replace the one of its name."""
        # The engine runs a reply's last <repl> block, so a cell writes [repl] instead.
        if type(yaml) is str:
            yaml = yaml.replace("[repl]", "<repl>").replace("[/repl]", "</repl>")
        check_writable(namespace, yaml, key_names)
        with app_file(library, "rw") as connection:
            stored, before = add_decomposition(
                connection, namespace, yaml, replace=True, allowed=may_write
            )
        done = "added" if before is None else "replaced"
        change = logged(
            done,
            before,
            yaml,
            what="decomposition",
            namespace=namespace,
            name=stored["name"],
        )
        return {**change, "problems": stored["problems"]}

    def add(name: str, yaml: str) -> dict[str, Any]:
        """Add namespace *name* with the fields *yaml* sets."""
        check_writable(name, yaml, key_names)
        with app_file(library, "rw") as connection:
            stored = add_namespace(connection, name, yaml, allowed=may_change)
        return logged(
            "added", None, stored, what="namespace", namespace=name, name=name
        )

    def change(name: str, yaml: str) -> dict[str, Any]:
        """Give namespace *name* the fields *yaml* sets in place of its own."""
        check_writable(name, yaml, key_names)
        with app_file(library, "rw") as connection:
            before, stored = change_namespace(
                connection, name, yaml, allowed=may_change
            )
        return logged(
            "changed", before, stored, what="namespace", namespace=name, name=name
        )

    def check_builds(name: str, block: dict[str, Any], source: str) -> None:
        """Raise unless the engine builds tool *name* from *source*, as a run would."""
        checks = tools_dir / CHECKS
        checks.mkdir(parents=True, exist_ok=True)
        # A folder of its own each time: the engine keeps each file it imported, by path.
        folder = Path(tempfile.mkdtemp(dir=checks))
        file = folder / tool_file(name)
        file.write_text(source, encoding="utf-8")
        try:
            make_tool(
                name,
                block | {"factory_from": str(file)},
                client=client,
                config_path=None,
            )
        except ToolBuildError as error:
            raise LibraryError(
                f"Tool {name} doesn't build, so it wasn't written: {error.detail}"
            ) from error
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def write_tool(
        name: str, source: str, factory: str, settings: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Add tool *name*, built by the function *factory* in the Python *source* with
        *settings* as its params, or replace the one of its name."""
        check_tool(name, source, factory, settings, key_names)
        # The grant and the locks first, before the tool's code runs; the build holds no
        # write lock, so they are checked again under it.
        with app_file(library, "ro") as connection:
            may_write_tool(connection, name)
        block = {
            "factory": factory,
            "factory_from": tool_file(name),
            **(settings or {}),
        }
        check_builds(name, block, source)
        with app_file(library, "rw") as connection:
            found = store_tool(
                connection,
                name,
                block,
                source,
                allowed=lambda locked: may_write_tool(locked, name),
            )
        before = None if found is None else tool_text(name, *found)
        after = tool_text(name, dump(block), source)
        done = "added" if found is None else "replaced"
        return logged(done, before, after, what="tool", name=name)

    tool = SimpleNamespace(
        show=show,
        write_decomposition=write_decomposition,
        add_namespace=add,
        change_namespace=change,
        write_tool=write_tool,
    )
    return Func(tool, description=LIBRARY_DESCRIPTION)
