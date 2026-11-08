"""The agent's own tools: the app gives every run these, besides the Library's.

The engine builds each from this file by its path (factory_from), in the worker, with only
what its block in the run's config holds: the app file's path, the chat, the run, the chat's
namespace and the variables that hold keys. Each opens the app file as it is, never making
a moved or deleted one afresh.

history reads the chat's earlier runs from the app file, read-only, as the page reads them:
runs.py checks each row, and folded() folds a run's events into its tree as fold.js does.
Each long text it returns is cut, so one huge output can't swamp the prompt; full() gives a
step whole.

page builds a page for the user from the same trees: each step or sub-agent it points to
is copied from the record, whole, as fold.js folds it, into the pages table; a page opened
is logged as page.open, which the worker records as a run event, once: opened again, it is
saved in its place.

library reads and writes the Library through library.py, with the panel's checks and
transactions; the locks are here, as the panel may still edit root, a namespace's repl and
reasoner, and which namespaces grant the app's own tools. It writes only in a run whose
chat's namespace grants it, so a sub-agent spawned into such a namespace from another chat
can't, and never a key. It guards the tool, not the file: a cell can open the file itself.
A tool it writes, or a custom tool a namespace it writes grants afresh, is built first as
another chat's run would build it: in a process of its own (tool_check.py), from an empty
folder. Each change is logged as library.change, which the worker records as a run event,
and changes() reads this chat's back, so the agent can write a before back.
"""

import copy
import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from contextlib import closing, contextmanager
from contextvars import copy_context
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jinja2
import structlog
import yaml
from deep_reasoner.namespaces import ROOT, NamespaceConfig, NamespaceRegistry
from deep_reasoner.primitives import Func
from deep_reasoner.v2.cli import V2Config

from deep_reasoning import tool_check
from deep_reasoning.library import (
    APP_TOOLS,
    CLAUDE_CODE,
    HISTORY_DESCRIPTION,
    LIBRARY_DESCRIPTION,
    LIBRARY_TOOL,
    PAGE_DESCRIPTION,
    TAKEN_NAMES,
    LibraryError,
    add_decomposition,
    add_namespace,
    built_grants,
    change_namespace,
    check_size,
    check_tool_block,
    config_and_registry,
    config_of,
    count,
    distrust,
    dump,
    fields_of,
    in_chain_order,
    literals,
    loaded,
    namespace_rows,
    parsed,
    parsed_python,
    read_library,
    reading,
    refuse_keys,
    run_config_and_registry,
    running_code,
    same_grants,
    sees,
    store_tool,
    taken,
    texts_in,
    tool_file,
    tool_rows,
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
    stored_text,
    update_page,
)
from deep_reasoning.runs import (
    CHANGE_FIELDS,
    PAGE_FIELDS,
    SPEC,
    chat_turns,
    raised_error,
    secrets_of,
    total_cost,
    withheld,
)
from deep_reasoning.tool_check import STOPS_EVERY_CHAT, refuse_unbuilt

log = structlog.get_logger()

Event = dict[str, Any]
Holder = dict[str, Any]
STARTED_WITH = ("node", "parent", "drive", "task", "namespace")
AGENT_FIELDS = (*STARTED_WITH, "status", "result")
ENDED_WITH = ("outcome", "answer", "detail", "cost", "seconds")
# An agent's time, to the tenth of a second, as the tree shows it.
SECONDS_DIGITS = 1
LOOKUP_FIELDS = ("node", "parent", "cell")
ROOT_LOCKED = (
    "root is locked: the library tool never writes it. The user edits it in the Library"
    " panel (top right): its YAML with Edit YAML in the Namespaces tab (its system_suffix"
    " for what every chat should do), its worked examples in the Decompositions tab. Or"
    " write into another namespace."
)
# Where the user changes the Library: a chat in a namespace that sees the library tool.
PICK_AN_EDITOR = jinja2.Template(
    "pick {% if editors %}{{ editors | join(' or ') }}{% else %}a namespace that grants"
    " library{% endif %} under the message box",
    undefined=jinja2.StrictUndefined,
)
NOT_THIS_CHAT = jinja2.Template(
    "The Library can be changed only in a chat whose namespace grants the library tool;"
    " {{ pick }}.",
    undefined=jinja2.StrictUndefined,
)
NOT_TEXT = (
    "The namespace's name should be text, and the YAML text or a dict of plain values."
)
# Where a brief list, or history, cut a text short.
CUT_MARK = re.compile(r"…\(\+[\d,]+ chars\)")
CUT_SHORT = (
    "What you wrote holds “…(+N chars)”, where a brief list cut a text short: write the"
    " whole text, from show(name), tools(name) or changes()."
)
# An example's cell tags, each on a line of its own, which a cell writes as placeholders,
# as a reply's last real <repl> is the one that runs.
REAL_TAGS = {"[repl]": "<repl>", "[/repl]": "</repl>"}
HOLDS_A_KEY = (
    "What you wrote holds a provider's key, and the Library file is shared, so keys never"
    " go in it. Take the key out."
)
# reasoner: claude_code would bill the user's subscription.
LOCKED_FIELDS = ("repl", "reasoner")
USERS_CHOICE = "the user's choice, in the Library panel's Namespaces tab (Edit YAML)."
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
CHECKS = "checks"
# What wasn't done when a tool doesn't build: written, or granted.
WRITTEN = "wasn't written"
GRANTED = f"wasn't granted ({STOPS_EVERY_CHAT}; fix it with write_tool first)"
# Any other REPL is a sandbox, and building a tool runs its code outside it.
LOCAL_REPL = "local"
# Told to the library tool's agents in a chat that can't write, such as a sub-agent's.
READ_ONLY_HERE = jinja2.Template(
    "\nIn this chat you may only read the Library: its namespace, {{ namespace }}, doesn't"
    " grant library, so the tool refuses every write, here and in any sub-agent. Asked"
    " for a change, don't try it: answer, with FinalAnswer, what you would change, and"
    " that the user should {{ pick }} and ask there; but root, which every chat builds"
    " on, only the user changes, in the Library panel (Namespaces, root, Edit YAML).",
    undefined=jinja2.StrictUndefined,
)
NOT_LOCAL = (
    "Writing a tool, or granting a custom tool, works only in a chat whose namespace runs"
    " the local REPL: building the tool to check it runs its code outside any sandbox."
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
NO_EARLIER = "This chat has no earlier messages: this is its first."
# history cuts each text it returns past this, so one huge output can't swamp the prompt;
# history.full gives a step whole.
MAX_VALUE_CHARS = 2_000
# And library.show() and tools() each text in their brief lists past this: a Library may
# hold hundreds of namespaces and examples, and show(name) or tools(name) gives one whole.
BRIEF_CHARS = 200
# And each list or dict nested deeper than this: no run of the app's nests one past a
# few levels, and a file from someone else may nest one thousands deep.
MAX_VALUE_DEPTH = 20
NESTED_TOO_DEEP = "…(nested too deep)"
# errors and find list at most this many rows, then one saying how many more.
MAX_FOUND = 50
# What find shows on each side of a match.
SNIPPET_CHARS = 60
NOT_FINDABLE = (
    "find takes the text to look for, as text (a str) of one character or more."
)
ENDED_BADLY = ("failed", "stopped", "exhausted")
# What find searches in each kind of step, as (kind, field).
SEARCHED = {
    "think": [("think", "text")],
    "cell": [("code", "code"), ("output", "output")],
}


@dataclass
class Tree:
    """A run's events folded: the root agent's drive, each sub-agent drive as a row, and
    each node's current drive (its holder) with its parent's node."""

    root: Holder = field(
        default_factory=lambda: {"node": None, "drive": None, "costs": [], "steps": []}
    )
    rows: list[Holder] = field(default_factory=list)
    model: Any = None
    nodes: dict[Any, tuple[Holder, Any]] = field(default_factory=dict)
    cells: dict[Any, dict[str, Any]] = field(default_factory=dict)
    # The drives each cell started, by the cell's id(), for a page's copy of it.
    started: dict[int, list[Holder]] = field(default_factory=dict)


def start_turn(tree: Tree, event: Event) -> None:
    tree.model = event.get("model")


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
    """A call's cost counts for its agent and every agent above it; a sub-agent's own
    call names the model it asked for on its row, when that isn't the run's."""
    node, seen = event.get("node"), set()
    row = holder_of(tree, node)
    model = event.get("model")
    own = event.get("own") is True and row is not None and row is not tree.root
    if own and isinstance(model, str) and model != tree.model:
        row["model"] = model
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
    "turn.start": start_turn,
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


def failed(step: Event) -> bool:
    return step["kind"] == "cell" and step["status"] == "failed"


def failed_cells(holder: Holder) -> int:
    return sum(map(failed, holder["steps"]))


def indexed(drives: list[Holder]) -> Iterator[tuple[int, Holder, Event]]:
    """Each step of *drives* as steps() lists them: its place from 0, its drive, and the
    step."""
    steps = ((row, step) for row in drives for step in row["steps"])
    return ((index, row, step) for index, (row, step) in enumerate(steps))


def turn_row(number: int, events: list[Event], tree: Tree) -> dict[str, Any]:
    start, end = events[0], events[-1]
    return {
        "turn": number,
        "message": start["prompt"],
        "namespace": start.get("namespace"),
        "model": start.get("model"),
        **{key: end.get(key) for key in ENDED_WITH},
        "agents": len(tree.rows),
        "failed": sum(map(failed_cells, [tree.root, *tree.rows])),
    }


def agent_row(row: Holder) -> dict[str, Any]:
    fields = {key: row[key] for key in AGENT_FIELDS}
    seconds = row["seconds"]
    return fields | {
        "seconds": None if seconds is None else round(seconds, SECONDS_DIGITS),
        "cost": total_cost(row["costs"]),
        "steps": len(row["steps"]),
        "failed": failed_cells(row),
    }


def cut(value: Any, depth: int = 0, limit: int = MAX_VALUE_CHARS) -> Any:
    """*value*, copied, with each text past *limit* characters cut there, saying how much
    is left out, and each list or dict past MAX_VALUE_DEPTH levels marked in its place."""
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]}…(+{len(value) - limit:,} chars)"
    if not isinstance(value, dict | list):
        return value
    if depth == MAX_VALUE_DEPTH:
        return NESTED_TOO_DEEP
    if isinstance(value, dict):
        return {name: cut(each, depth + 1, limit) for name, each in value.items()}
    return [cut(each, depth + 1, limit) for each in value]


def error_of(output: Any) -> str | None:
    """What a failed cell raised, as its traceback's exception lines; else, as in a record
    from someone else, its output's last line."""
    if not isinstance(output, str):
        return None
    lines = output.strip().splitlines()
    return raised_error(output) or (lines[-1] if lines else None)


def capped(found: list[dict[str, Any]], narrower: str) -> list[dict[str, Any]]:
    """*found*, at most MAX_FOUND rows of it, then a row saying how many more were left
    out and how to list them: *narrower*."""
    more = len(found) - MAX_FOUND
    if more <= 0:
        return found
    note = f"{more:,} more not listed: {narrower}"
    return [*found[:MAX_FOUND], {"more": more, "note": note}]


def ending(drive: Holder) -> str:
    """How an agent's drive that didn't answer ended, in words."""
    why = drive["result"]
    return f"It ended {drive['status']}" + (f": {why}" if why else ".")


def snippet(text: str, match: re.Match) -> str:
    start, end = max(match.start() - SNIPPET_CHARS, 0), match.end() + SNIPPET_CHARS
    return f"{'…' * (start > 0)}{text[start:end]}{'…' * (end < len(text))}"


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


def written[T](library: str, write: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """What *write* returns, given the app file opened to write and *args*."""
    with app_file(library, "rw") as connection:
        return write(connection, *args, **kwargs)


def stop_proof[T](work: Callable[[], T]) -> T:
    """What *work* returns, worked out on a thread of its own, in this thread's context.
    The engine stops a cell by raising StopRequested in the cell's thread, once, at its
    next bytecode: one that arrives during a commit, in C, is raised as the commit
    returns. Here it is raised before the other thread takes *work* up, which it then
    never does, or once *work* is done; so a Stop never falls between a write and the
    event that logs it. The server's kill, 5 s after a Stop, still could, were a write
    still going then and the kill to land between its commit and its log."""
    future = Future()

    def take_up() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(work())
        # Handed on whole, as result() raises it in the caller's thread.
        except BaseException as error:  # noqa: BLE001
            future.set_exception(error)

    thread = threading.Thread(target=copy_context().run, args=(take_up,))
    try:
        thread.start()
        thread.join()
    except BaseException:
        if not future.cancel():
            thread.join()
        raise
    return future.result()


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
            raise ValueError(NO_EARLIER)
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

    def located(self, turn: int, node: Any, index: int) -> tuple[int, Holder, Event]:
        """Step *index* of message *turn*'s top agent, or of sub-agent *node*, as steps()
        lists them: its place from 0, the drive it is in, and the step."""
        if type(index) is not int:
            raise ValueError(NOT_AN_INDEX)
        steps = list(indexed(self.drives(turn, node)))
        if not -len(steps) <= index < len(steps):
            who = "top agent" if node is None else f"agent {node!r}"
            raise ValueError(
                f"Message {turn}'s {who} has {count(len(steps), 'step')}, numbered from"
                f" 0: there is no step {index}."
            )
        return steps[index]

    def numbered(self, turn: int | None) -> list[int]:
        """Message *turn*, once it is checked, or every earlier one when None."""
        if turn is not None:
            self.tree(turn)
            return [turn]
        if not self._runs:
            raise ValueError(NO_EARLIER)
        return list(range(1, len(self._runs) + 1))

    def agents_of(self, turn: int) -> Iterator[tuple[Any, list[Holder]]]:
        """Message *turn*'s agents as drives() reaches them, each as its node and drives:
        the top agent first, as node None, its drive with the task, namespace, status and
        result a sub-agent's has. A sub-agent's row named None or the root's node, as a file from
        someone else may hold, is the top agent's to drives(), so it is left out."""
        events, tree = self._runs[turn - 1]
        start, end = events[0], events[-1]
        outcome = end.get("outcome")
        top = tree.root | {
            "task": start["prompt"],
            "namespace": start.get("namespace"),
            "status": "done" if outcome == "answered" else outcome,
            "result": end.get("answer") or end.get("detail"),
        }
        yield None, [top]
        sub_agents: dict[Any, list[Holder]] = {}
        for row in tree.rows:
            if row["node"] is not None and row["node"] != tree.root["node"]:
                sub_agents.setdefault(row["node"], []).append(row)
        yield from sub_agents.items()

    def turns(self) -> list[dict[str, Any]]:
        """This chat's earlier messages, numbered from 1, with how each ended."""
        return cut(
            [turn_row(number, *each) for number, each in enumerate(self._runs, 1)]
        )

    def agents(self, turn: int) -> list[dict[str, Any]]:
        """Each sub-agent drive of message *turn*."""
        return cut([agent_row(row) for row in self.tree(turn).rows])

    def steps(self, turn: int, node: Any = None) -> list[dict[str, Any]]:
        """The thoughts, cells, Library changes and pages of message *turn*'s top agent,
        or of sub-agent *node*."""
        # A copy: the folded runs are kept for the next call.
        return cut([step for _, _, step in indexed(self.drives(turn, node))])

    def full(self, turn: int, node: Any, index: int) -> dict[str, Any]:
        """steps(turn, node)[index], nothing cut; a ValueError when it nests too deep to
        copy, as a record in a shared file may."""
        step = self.located(turn, node, index)[2]
        try:
            return copy.deepcopy(step)
        except RecursionError as error:
            raise ValueError(TOO_DEEP) from error

    def errors(self, turn: int | None = None) -> list[dict[str, Any]]:
        """What went wrong in message *turn*, or in every earlier one: each failed cell of
        any of its agents, the top one included, and each drive that didn't answer."""
        found = [
            each for number in self.numbered(turn) for each in self.failures(number)
        ]
        return cut(capped(found, "pass a turn to list only that message's."))

    def failures(self, turn: int) -> Iterator[dict[str, Any]]:
        """Message *turn*'s failed cells and drives that didn't answer, agent by agent."""
        for node, drives in self.agents_of(turn):
            at = {"turn": turn, "node": node}
            yield from (
                at
                | {"namespace": row["namespace"], "task": row["task"], "index": index}
                | {"error": error_of(step["output"]), "code": step["code"]}
                for index, row, step in indexed(drives)
                if failed(step)
            )
            yield from (
                at
                | {"namespace": row["namespace"], "task": row["task"], "index": None}
                | {"error": ending(row), "code": None}
                for row in drives
                if row["status"] in ENDED_BADLY
            )

    def find(self, text: str, turn: int | None = None) -> list[dict[str, Any]]:
        """Where *text* is, ignoring case, in message *turn* or every earlier one: in any
        agent's tasks, thoughts, code, outputs and results."""
        if type(text) is not str or not text:
            raise ValueError(NOT_FINDABLE)
        pattern = re.compile(re.escape(text), re.IGNORECASE)
        found = [
            {"turn": number, "node": node, "index": index, "kind": kind}
            | {"snippet": snippet(value, match)}
            for number in self.numbered(turn)
            for node, index, kind, value in self.texts(number)
            if isinstance(value, str) and (match := pattern.search(value))
        ]
        return cut(capped(found, "pass a turn, or longer text, to list fewer."))

    def texts(self, turn: int) -> Iterator[tuple[Any, int | None, str, Any]]:
        """Each text of message *turn* that find searches, as its agent's node, its step's
        index (None for a task or result), its kind and the text."""
        for node, drives in self.agents_of(turn):
            yield from ((node, None, "task", row["task"]) for row in drives)
            yield from (
                (node, index, kind, step[key])
                for index, _, step in indexed(drives)
                for kind, key in SEARCHED.get(step["kind"], [])
            )
            yield from ((node, None, "result", row["result"]) for row in drives)


def build_history(client: Any, params: dict[str, Any]) -> Func:
    """The history tool of run *run* of *chat*, in the app file *library*, as its block
    in the run's config names them."""
    history = History(params["library"], params["chat"], params["run"])
    methods = SimpleNamespace(
        turns=history.turns,
        agents=history.agents,
        steps=history.steps,
        errors=history.errors,
        find=history.find,
        full=history.full,
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
        add: Callable[[str, str], str],
        update: Callable[[str, str], None],
    ) -> None:
        self._title = checked_title(title, secrets)
        self._blocks = []
        self._history, self._secrets = history, secrets
        self._add_page, self._update_page = add, update
        self._chars = len(json.dumps([]))
        # Its id once opened, so opening it again saves it in its place.
        self._page = None

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
        index, row, step = self._history.located(turn, node, index)
        block = {
            "type": "step",
            "turn": turn,
            "node": node,
            "index": index,
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
        """Save the page, which the chat links under the answer; opened again, it is saved
        as it stands then, in its place, and linked no second time. What it saves is
        checked again, as a cell may have set the page's state itself."""
        title = checked_title(self._title, self._secrets)
        if not self._blocks:
            raise ValueError(EMPTY)
        blocks = stored_text(keys_withheld(self._blocks, self._secrets))
        if blocks is None:
            raise ValueError(NOT_SAVED)
        if self._page is not None:
            self._update_page(self._page, blocks)
            return (
                f'Saved the page "{title}" again, as it stands now; the chat still'
                " links it."
            )
        stop_proof(partial(self._open_anew, title, blocks))
        return f'Opened the page "{title}"; the chat links it under your answer.'

    def _open_anew(self, title: str, blocks: str) -> None:
        """Save the page anew, as this page, and log page.open."""
        self._page = self._add_page(title, blocks)
        log.info("page.open", page=self._page, title=title)

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

    def add(title: str, blocks: str) -> str:
        """Save a new page of this run's, its *blocks* as JSON; its id."""
        # The agent whose cell opens the page, as its page.open event names it.
        node = structlog.contextvars.get_contextvars().get("agent_id")
        return written(library, add_page, chat, run, node, title, blocks)

    def update(page: str, blocks: str) -> None:
        """Save page *page* of this run's again, with *blocks*, as JSON."""
        written(library, update_page, chat, run, page, blocks)

    def new(title: str) -> Page:
        """A page titled *title*, saved when its open() is called."""
        return Page(title, history, secrets, add, update)

    return Func(SimpleNamespace(new=new), description=PAGE_DESCRIPTION)


def check_no_key(texts: list[str], key_names: list[str]) -> None:
    """Raise if any of *texts* holds the value of a key variable in *key_names*."""
    refuse_keys(texts, secrets_of(os.environ, key_names), HOLDS_A_KEY)


def with_real_tags(text: str) -> str:
    """The decomposition written as YAML *text* with each line of an assistant turn that
    is a placeholder tag alone, [repl] or [/repl], written as the real one; *text* as it
    is when it has none, or doesn't read as a decomposition, which the store refuses."""
    try:
        decomposition = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    messages = (
        decomposition.get("messages") if isinstance(decomposition, dict) else None
    )
    changed = False
    for turn in messages if isinstance(messages, list) else []:
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        content = turn.get("content")
        if isinstance(content, str):
            lines = content.splitlines(keepends=True)
            turn["content"] = "".join(map(tag_written, lines))
            changed = changed or turn["content"] != content
    return dump(decomposition) if changed else text


def tag_written(line: str) -> str:
    """*line*, written as the real tag when it is a placeholder tag alone, its indent kept."""
    tag = line.strip()
    return line.replace(tag, REAL_TAGS[tag]) if tag in REAL_TAGS else line


def yaml_text(value: Any) -> str:
    """*value*, YAML text or a dict of plain values, as YAML text, from its first line
    that isn't blank."""
    if type(value) is dict and plain(value):
        return dump(value)
    # Exactly str: a subclass could compare unequal to "root" and still name it.
    if type(value) is not str:
        raise LibraryError(NOT_TEXT)
    return value.lstrip("\n")


def check_whole(texts: Iterable[str]) -> None:
    """Raise if any of *texts* holds the mark of a text a brief list cut short."""
    if any(CUT_MARK.search(text) for text in texts):
        raise LibraryError(CUT_SHORT)


def check_writable(namespace: str, text: str, key_names: list[str]) -> None:
    """Raise unless the tool may write the YAML *text* for the namespace named
    *namespace*."""
    if type(namespace) is not str:
        raise LibraryError(NOT_TEXT)
    if namespace == ROOT:
        raise LibraryError(ROOT_LOCKED)
    check_size(text, "The YAML")
    check_whole([text])
    # Parsed too, as an escape such as "\x73k-" spells a key the raw text doesn't hold.
    check_no_key([namespace, text, *texts_in(loaded(text))], key_names)


def code_problems(text: str) -> list[str]:
    """Each assistant turn of the decomposition written as YAML *text* whose code, the
    <repl> block the engine runs, isn't valid Python."""
    problems = []
    for number, turn in enumerate(parsed(text).messages, start=1):
        if turn.role != "assistant":
            continue
        try:
            parsed_python(
                running_code(turn.content), f"Turn {number} (assistant)'s code"
            )
        except LibraryError as error:
            problems.append(str(error))
    return problems


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
    if taken(name):
        raise LibraryError(
            f'"{name}" is taken: it is a Python keyword, a built-in or app tool, or a name'
            f" the REPL binds itself ({', '.join(TAKEN_NAMES)}). Pick another."
        )
    check_tool_block(name, {"factory": factory, "factory_from": tool_file(name)})
    check_size(source, "The source")
    check_size(dump(settings), "The settings as YAML")
    check_whole([source, *texts_in(settings)])
    # texts holds the source itself.
    check_no_key(
        [*texts, *texts_in(settings), *literals(parsed_python(source))], key_names
    )


def check_tool_locks(config: V2Config, registry: NamespaceRegistry, name: str) -> None:
    """Raise if the tool may not write tool *name*: root grants it, or the engine builds
    it from its block alone, which the tool couldn't write back."""
    if name in registry.config(ROOT).tools:
        raise LibraryError(
            f"root grants {name}, so every chat would run the code you write: only the"
            " user may replace it, once they take it out of root's tools in the Library"
            " panel's Namespaces tab (Edit YAML)."
        )
    block = config.tools.get(name)
    if block is not None and "factory_from" not in block:
        raise LibraryError(
            f"{name} is built by the engine's own {block.get('factory', name)} factory"
            " from its block alone, so it has no Python for write_tool to write back:"
            " only the user may replace it, by importing a config that defines it"
            " (deep-reasoning import). Write a tool of another name instead."
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


def field_changes(before: str, after: str) -> tuple[dict[str, Any], list[str]]:
    """Each field a namespace's own YAML *after* sets otherwise than *before* does, with
    its value in each (None when unset), and the names of those it leaves as they were."""
    old, new = fields_of(before), fields_of(after)
    keys = dict.fromkeys([*old, *new])
    changed = {
        key: {"before": old.get(key), "after": new.get(key)}
        for key in keys
        if old.get(key) != new.get(key)
    }
    return changed, [key for key in keys if key not in changed]


def in_brief(row: dict[str, Any], registry: NamespaceRegistry) -> dict[str, Any]:
    """A namespace as show() lists it: what it sets itself and the tools its agents get,
    and each of its decompositions with how many problems it has, each text cut short."""
    brief = {
        "name": row["name"],
        "depth": row["depth"],
        "sets": fields_of(row["yaml"]),
        "tools": list(registry.resolve(row["name"]).tools),
        "decompositions": [
            {"name": each["name"], "problems": len(each["problems"])}
            for each in row["decompositions"]
        ],
    }
    return cut(brief, limit=BRIEF_CHARS)


def tool_entries(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every tool, as the panel's Tools tab lists them, with its settings and a custom
    tool's Python source."""
    read = read_library(connection)
    config, registry = run_config_and_registry(config_of(read))
    sources = {name: source for name, (_, source) in read["tools"].items()}
    rows = tool_rows(config, registry, in_chain_order(registry), sources)
    return [
        {
            "name": row["name"],
            "kind": row["badge"] or "custom",
            "factory": row["factory"],
            "settings": {
                key: value
                for key, value in config.tools.get(row["name"], {}).items()
                if key not in OWN_KEYS
            },
            "granted_in": row["granted_in"],
            "description": row["description"],
            "source": row["source"],
        }
        for row in rows
    ]


def named(rows: list[dict[str, Any]], name: Any, noun: str, lister: str) -> Any:
    """The one of *rows* named *name*; a LibraryError saying where they are listed when
    none is."""
    found = [row for row in rows if row["name"] == name]
    if not found:
        raise LibraryError(f'The Library has no {noun} "{name}": {lister} lists them.')
    return found[0]


def tool_fields(name: Any, text: Any) -> Any:
    """Tool *name*'s row, as tool_text writes it, as write_tool takes it: {source,
    factory, settings}; *text* as it is when it isn't one, as any cell may log a change
    and a shared file hold one."""
    if type(name) is not str or type(text) is not str:
        return text
    found = re.search(rf"^# {re.escape(name)}\.py\n", text, re.MULTILINE)
    try:
        block = yaml.safe_load(text[: found.start()]) if found else None
    except (yaml.YAMLError, RecursionError):
        return text
    if type(block) is not dict:
        return text
    settings = {key: value for key, value in block.items() if key not in OWN_KEYS}
    return {
        "source": text[found.end() :],
        "factory": block.get("factory", name),
        "settings": settings or None,
    }


def as_written(event: Event) -> dict[str, Any]:
    """A library.change *event*'s fields: a namespace's with each field it changed, and a
    tool's before and after as write_tool takes them."""
    change = {key: event.get(key) for key in CHANGE_FIELDS}
    if change["what"] == "namespace":
        before, after = change["before"], change["after"]
        change["changed"] = field_changes(before or "", after)[0]
    if change["what"] == "tool":
        for key in ("before", "after"):
            change[key] = tool_fields(change["name"], change[key])
    return change


def chat_changes(library: str, chat: str, turn: Any) -> list[dict[str, Any]]:
    """The Library changes of *chat*'s message *turn*, or of every message of its, this
    one's so far included, each with its turn and message."""
    with app_file(library, "ro") as connection:
        turns = chat_turns(connection, chat)
    if turn is not None and (type(turn) is not int or not 1 <= turn <= len(turns)):
        raise ValueError(
            f"There is no message {turn!r}: this chat's messages are numbered 1 to"
            f" {len(turns)}, this one last."
        )
    found = []
    for number, each in enumerate(turns, start=1):
        if turn not in (None, number):
            continue
        events = each["events"]
        at = {"turn": number, "message": cut(events[0]["prompt"])}
        found += [
            at | as_written(event)
            for event in events
            if event["kind"] == "library.change"
        ]
    return found


def check_builds(
    tools: dict[str, tuple[dict[str, Any], str]], tools_dir: Path, done: str
) -> None:
    """Raise unless each of *tools*, by name its block and Python source, builds as a run
    of this run's config, in its folder beside *tools_dir*, would build it, in this
    worker's environment. *done* says what wasn't done when one doesn't."""
    if not tools:
        return
    config = tool_check.config_of(tools_dir.parent / SPEC)
    refuse_unbuilt(tools, config, os.environ, tools_dir / CHECKS, done)


def build_library(client: Any, params: dict[str, Any]) -> Func:
    """The library tool, over the app file *library* its block in the run's config names,
    for a run of a chat in *namespace*, whose worker holds keys in *key_names*; it builds
    the tools it checks in the run's *tools_dir*."""
    library, chat_namespace = params["library"], params["namespace"]
    key_names, tools_dir = params["key_names"], Path(params["tools_dir"])
    editors = params.get("editors", [])
    pick = PICK_AN_EDITOR.render(editors=editors)

    def show(name: str | None = None) -> Any:
        """Every namespace in brief, or namespace *name* as the Library panel shows it."""
        with app_file(library, "ro") as connection, reading(connection):
            read = read_library(connection)
            config, registry = run_config_and_registry(config_of(read))
            # Not those the engine can't read, as a shared file may hold: no agent runs there.
            rows = [
                row
                for row in namespace_rows(read, config, registry)
                if row["name"] in registry.names
            ]
        if name is None:
            return [in_brief(row, registry) for row in rows]
        return named(rows, name, "namespace", "show()")

    def tools(name: str | None = None) -> Any:
        """Every tool in brief, or tool *name* whole, with its source."""
        with app_file(library, "ro") as connection, reading(connection):
            rows = tool_entries(connection)
        if name is None:
            return cut(rows, limit=BRIEF_CHARS)
        return named(rows, name, "tool", "tools()")

    def changes(turn: int | None = None) -> list[dict[str, Any]]:
        """This chat's Library changes, in message *turn* or in every one."""
        return chat_changes(library, params["chat"], turn)

    def check_grant(registry: NamespaceRegistry) -> None:
        """Raise unless this run's chat is in a namespace that grants the library tool, as
        the Library stands under the write lock: a grant taken away meanwhile counts."""
        granted = chat_namespace in registry.names and (
            sees(registry, chat_namespace, LIBRARY_TOOL)
        )
        if not granted:
            raise LibraryError(NOT_THIS_CHAT.render(pick=pick))

    def check_local(registry: NamespaceRegistry) -> None:
        """Raise unless this run's chat runs the local REPL: a build runs a tool's code
        outside any sandbox."""
        if registry.resolve(chat_namespace).repl.type != LOCAL_REPL:
            raise LibraryError(NOT_LOCAL)

    def may_write(connection: sqlite3.Connection) -> None:
        check_grant(config_and_registry(connection)[1])

    def locks(
        connection: sqlite3.Connection,
        namespace: NamespaceConfig,
        afresh: dict[str, tuple[dict[str, Any], str]],
    ) -> None:
        """Raise unless the tool may write *namespace*, as the Library stands, building
        the custom tools it grants *afresh* if any."""
        config, registry = config_and_registry(connection)
        check_grant(registry)
        check_locks(config, registry, namespace)
        if afresh:
            check_local(registry)

    def may_write_tool(connection: sqlite3.Connection, name: str) -> None:
        config, registry = config_and_registry(connection)
        check_grant(registry)
        check_local(registry)
        check_tool_locks(config, registry, name)

    def grants_built(
        name: str, text: str, adding: bool, merge: bool = False
    ) -> Callable[[sqlite3.Connection, NamespaceConfig], None]:
        """The write's *allowed* for namespace *name* as the YAML *text* makes it, once
        the custom tools it grants afresh are built (built_grants())."""
        build = partial(check_builds, tools_dir=tools_dir, done=GRANTED)
        with app_file(library, "ro") as connection:
            built = built_grants(
                connection, name, text, build, adding, merge=merge, locks=locks
            )
        return partial(same_grants, built, locks)

    def write_decomposition(namespace: str, yaml: Any) -> dict[str, Any]:
        """Add the decomposition *yaml* to *namespace*, or replace the one of its name."""
        text = yaml_text(yaml)
        # A text with a real tag was read back, not typed, so its [repl] is the example's.
        if "<repl>" not in text:
            text = with_real_tags(text)
        check_writable(namespace, text, key_names)

        def write() -> tuple[dict[str, Any], Event]:
            stored, before = written(
                library,
                add_decomposition,
                namespace,
                text,
                replace=True,
                allowed=may_write,
            )
            done = "added" if before is None else "replaced"
            change = logged(
                done,
                before,
                text,
                what="decomposition",
                namespace=namespace,
                name=stored["name"],
            )
            return stored, change

        stored, change = stop_proof(write)
        broken = code_problems(text)
        return {**change, "problems": stored["problems"]} | (
            {"code_problems": broken} if broken else {}
        )

    def add(name: str, yaml: Any) -> dict[str, Any]:
        """Add namespace *name* with the fields *yaml* sets."""
        text = yaml_text(yaml)
        check_writable(name, text, key_names)
        allowed = grants_built(name, text, adding=True)

        def write() -> Event:
            stored = written(library, add_namespace, name, text, allowed=allowed)
            return logged(
                "added", None, stored, what="namespace", namespace=name, name=name
            )

        return stop_proof(write)

    def change(name: str, yaml: Any, replace: bool = False) -> dict[str, Any]:
        """Give namespace *name* the fields *yaml* sets, keeping the others, or with
        *replace* in place of all its own."""
        text = yaml_text(yaml)
        check_writable(name, text, key_names)
        merge = not replace
        allowed = grants_built(name, text, adding=False, merge=merge)
        at = {"what": "namespace", "namespace": name, "name": name}

        def write() -> tuple[str, str, Event | None]:
            before, stored = written(
                library, change_namespace, name, text, allowed=allowed, merge=merge
            )
            done = None if stored == before else logged("changed", before, stored, **at)
            return before, stored, done

        before, stored, done = stop_proof(write)
        if done is None:
            return {"done": "unchanged", **at}
        changed, kept = field_changes(before, stored)
        return done | {"changed": cut(changed), "kept": kept}

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
        check_builds({name: (block, source)}, tools_dir, WRITTEN)

        def write() -> Event:
            allowed = partial(may_write_tool, name=name)
            before = written(library, store_tool, name, block, source, allowed=allowed)
            after = tool_text(name, dump(block), source)
            done = "added" if before is None else "replaced"
            return logged(done, before, after, what="tool", name=name)

        return stop_proof(write)

    tool = SimpleNamespace(
        show=show,
        tools=tools,
        changes=changes,
        write_decomposition=write_decomposition,
        add_namespace=add,
        change_namespace=change,
        write_tool=write_tool,
    )
    description = LIBRARY_DESCRIPTION
    if chat_namespace not in editors:
        description += READ_ONLY_HERE.render(namespace=chat_namespace, pick=pick)
    return Func(tool, description=description)
