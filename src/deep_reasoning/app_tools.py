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
and kept in the app file's record of changes, beside the panel's and imports', which
changes() reads, so the agent can say what changed, whoever changed it, and write a before
back.
"""

import ast
import copy
import io
import json
import math
import os
import re
import sqlite3
import sys
import textwrap
import threading
import tokenize
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from contextlib import closing, contextmanager
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import cached_property, partial, wraps
from importlib.util import find_spec
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
    APP_TOOLS_MODULE,
    CLAUDE_CODE,
    HISTORY_DESCRIPTION,
    LIBRARY_DESCRIPTION,
    LIBRARY_TOOL,
    LOCAL_REPL,
    MAX_YAML_CHARS,
    PAGE_DESCRIPTION,
    TAKEN_NAMES,
    LibraryError,
    add_decomposition,
    add_namespace,
    built_grants,
    by_chat,
    change_namespace,
    check_size,
    check_tool_block,
    config_and_registry,
    config_of,
    count,
    cut_text,
    distrust,
    dump,
    expanded_shape,
    fields_of,
    in_chain_order,
    is_left_out_note,
    left_out_note,
    literals,
    loaded,
    namespace_rows,
    parsed_python,
    read_changes,
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
    tool_name_problem,
    tool_rows,
    tool_text,
    without_impossible_hand_offs,
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
    FORWARD,
    LEFT_OUT,
    LEFT_OUT_FIELDS,
    PAGE_FIELDS,
    SPEC,
    chat_runs,
    chat_turns,
    first_loaded,
    raised_error,
    secrets_of,
    started,
    total_cost,
    withheld,
)
from deep_reasoning.tool_check import STOPS_EVERY_CHAT, refuse_unbuilt

# Named as the module, though the engine loads it by its path under a name of its own:
# the worker keeps the app's logs, not the agent's, out of cells' output.
log = structlog.get_logger(APP_TOOLS_MODULE)

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
# A {name} field, not {{ … }}, in a string written without f stays as typed, so a worked
# example passing one passes the braces, not the variable.
FIELD = re.compile(r"(?<!\{)\{([A-Za-z_]\w*)[^{}]*\}(?!\})")
FORMAT_METHODS = {"format", "format_map"}
UNFILLED = jinja2.Template(
    "{{ what }} has {{ '{' }}{{ name }}{{ '}' }} in a string without f, which stays as"
    " typed: an f-string puts {{ name }}'s value there.",
    undefined=jinja2.StrictUndefined,
)
# Where a brief list, or history, cut a text, list or dict short.
CUT_MARK = re.compile(r"…\(\+[\d,]+ (?:chars|more items)\)")
CUT_SHORT = (
    "What you wrote holds “…(+N chars)” or “…(+N more items)”, where a text or list was"
    " cut short: write the whole of it, from show(name, whole=True), tools(name) or"
    " changes()."
)
# An example's cell tags, each on a line of its own, as a cell may type them, as
# placeholders: a reply's last real <repl> is the one that runs.
REAL_TAGS = {"[repl]": "<repl>", "[/repl]": "</repl>"}
LONE_PLACEHOLDER = re.compile(r"^([ \t]*)\[(/?)repl\]([ \t]*)$", re.MULTILINE)
# How every text the tool gives writes a repl tag, which typed into a cell would end the
# cell there, and how every text it takes is read back: as a mark nobody types. A typed
# placeholder, which a write takes for a tag, and the mark's own bracket are marked too,
# so a text given and written back is the same to the byte.
MARKS = {
    "<repl>": "⟦repl⟧",
    "</repl>": "⟦/repl⟧",
    "[repl]": "⟦[repl]⟧",
    "[/repl]": "⟦[/repl]⟧",
    "⟦": "⟦⟦",
}
UNMARKS = {mark: text for text, mark in MARKS.items()}
TO_MARK = re.compile("|".join(map(re.escape, MARKS)))
TO_UNMARK = re.compile("|".join(map(re.escape, UNMARKS)))
# The tags the engine reads as a turn's parts: inside a turn's text, one would end its
# part there or start another.
TAG = re.compile(r"</?(?:repl|think|observation)>", re.IGNORECASE)
TAG_IN_TURN = jinja2.Template(
    "Turn {{ number }}'s {{ field }} holds {{ tag }}, a tag the engine reads, which would"
    " end the turn's part there or start another: type no tags in turns, as the tool"
    " writes them. To show one in a text, write {{ placeholder }} instead.",
    undefined=jinja2.StrictUndefined,
)
HOLDS_A_KEY = (
    "What you wrote holds a provider's key, and the Library file is shared, so keys never"
    " go in it. Take the key out."
)
# Why the user sets each, in the panel, and the tool never does.
LOCKED_FIELDS = {
    "repl": (
        "where a namespace's agents run their Python (the local REPL, unsandboxed, or a"
        " sandbox, as repl: {type: daytona_sandbox}) is the user's to set, even where"
        " show(name) says the run settings set it"
    ),
    "reasoner": (
        "reasoner: claude_code runs a namespace's agents on Claude Code, billed to the"
        " user's Claude subscription, so its reasoner is the user's to set"
    ),
}
USERS_CHOICE = "the user's choice, in the Library panel's Namespaces tab (Edit YAML)."
# A var's or tool setting's name reads as a secret's when one of these, or its plural, is
# one of its word parts, or ends it after a word saying what it opens (apikey, authtoken),
# but not when it starts a longer word (tokenizer, keywords) or ends another (hotkey).
SECRET_WORDS = "(?:key|token|secret|password)s?"
SECRET_PART = re.compile(SECRET_WORDS, re.IGNORECASE)
SECRET_ENDING = re.compile(
    rf"(?:api|auth|access|private|client|session|refresh|bearer|secret|pass){SECRET_WORDS}$",
    re.IGNORECASE,
)
WORD_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
# And a value holding a run of a key's characters this long, whose letters and digits
# switch between lowercase, uppercase and digits this often, or for this share of them,
# reads as a key: random, as no id, date, path, model name or label is.
KEY_LENGTH = 16
KEY_RUN = re.compile(rf"[A-Za-z0-9+/=_-]{{{KEY_LENGTH},}}")
ONE_CASE_OR_DIGITS = re.compile(r"[a-z]+|[A-Z]+|[0-9]+")
MIN_SWITCHES = 10
MIN_SWITCH_SHARE = 0.4
# Unless the name ends in env and the value is an environment variable's name.
NAMES_A_VARIABLE = "env"
VARIABLE_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
ABSENT = object()
SECRET_LIKE = jinja2.Template(
    "{{ field }} may not set {{ names | join(', ') }} to what reads as a secret: named"
    " like one (key, token, secret, password) and shaped like a key, it would be saved in"
    " the Library file, which is shared with every chat and copied whole with the file."
    " Keep a secret in the shell that"
    " starts the app (export {{ variable }}=… before uv run deep-reasoning) and read it in"
    ' a tool with os.environ["{{ variable }}"]; a var or setting may name that variable'
    " instead, as {{ variable | lower }}_env: {{ variable }}. If it holds no secret, name"
    " it without those words.",
    undefined=jinja2.StrictUndefined,
)
# The variable a refusal suggests for a secret whose name makes no variable's.
ANY_VARIABLE = "SECRET"
# write_decomposition's turns, each in one of these shapes.
NOT_TURNS = (
    'name should be text, and turns a list of dicts, each one of {"user": text} (the'
    ' task, or a later message), {"think": text, "code": text} (a step: its reasoning'
    ' and the Python it runs, with no tags) and {"observation": text} (what the step'
    " before printed)."
)
STEP = ("think", "code")
TWO_FORMS = (
    "write_decomposition takes write_decomposition(namespace, name, turns), for a worked"
    " example written as turns, or write_decomposition(namespace, yaml), for one written"
    " as YAML: name the example once, and only with its turns."
)
NOT_A_TOOL = (
    "name, source and factory should be text, and settings a dict of plain values (text,"
    " numbers, True, False, None, and lists and dicts of them keyed by text), or None."
)
PLAIN = (str, int, float, bool, type(None))
# Deeper settings are refused, as YAML writes them by recursion.
MAX_NESTING = 20
# A run hands the factory every key of the tool's block but these.
OWN_KEYS = ("factory", "factory_from")
SETS_OWN_KEYS = "settings may not hold factory or factory_from: write_tool sets them."
CHECKS = "checks"
# What a write that leaves a text as it was says, so the model never reports it done.
ALREADY_HELD = (
    "already held what you gave (a .replace(...) that matches nothing gives a text back"
    " as it was)"
)
NOTHING_CHANGED = f"nothing changed: it {ALREADY_HELD}"
# What wasn't done when a tool doesn't build: written, or granted.
WRITTEN = "wasn't written"
GRANTED = f"wasn't granted ({STOPS_EVERY_CHAT}; fix it with write_tool first)"
# What a tool couldn't import, as a build's refusal says it, and what the user does.
MISSING_MODULE = re.compile(r"No module named '([^']+)'")
NOT_INSTALLED = jinja2.Template(
    "The app's Python has no {{ module }}: the user adds it by restarting the app as `uv"
    " run --with {{ module }} deep-reasoning` (with the package's name on PyPI, where it"
    " differs), then asks again; never install it from a cell.",
    undefined=jinja2.StrictUndefined,
)
IN_FOLDER_ONLY = jinja2.Template(
    "{{ module }} is in this chat's folder only, which another chat's run doesn't have:"
    " put what the tool needs from it into the tool's source.",
    undefined=jinja2.StrictUndefined,
)
# Told to the library tool's agents in a chat that can't write, such as a sub-agent's.
READ_ONLY_HERE = jinja2.Template(
    "\nIn this chat you may only read the Library: its namespace, {{ namespace }}, doesn't"
    " grant library, so the tool refuses every write, here and in any sub-agent. Asked"
    " for a change, don't try it: answer, with FinalAnswer, what you would change, in"
    " the user's words, and that the user should {{ pick }} and ask there. A lasting wish"
    ' ("always remember X", "from now on") is for a helper, which you can\'t make here:'
    " say to {{ pick }} and ask for a helper that remembers it, or to add it to one of"
    " theirs. Say nothing of root, which every chat builds on and only the user changes"
    " in the Library panel, unless they ask for every chat in so many words. This"
    " replaces any line above on wishes.",
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
# And show(name) each text past this, unless asked for it whole: a namespace's own YAML
# and worked examples fit, and a shared file's may hold a million characters each.
SHOWN_CHARS = 20_000
# And each list or dict nested deeper than this: no run of the app's nests one past a
# few levels, and a file from someone else may nest one thousands deep.
MAX_VALUE_DEPTH = 20
NESTED_TOO_DEEP = "…(nested too deep)"
# And all a value holds past this many characters: no run of the app's returns one near
# it, and a shared file's aliases can make millions of short texts.
MAX_TOTAL_CHARS = 1_000_000
# What is said in a list, or as a key of a dict, of the items left out past it.
MORE = "…(+{:,} more items)"
# errors and find list at most this many rows, then one saying how many more.
MAX_FOUND = 50
# A change whose text was left out, as a row the Library couldn't read then: its text is
# a note saying why (left_out_note()), which no write takes back.
CANT_WRITE_BACK = (
    "Its {} couldn't be read when it was made, so the text is a note saying why: it"
    " can't be written back as is."
)
# The oldest id a page of changes read, whether its row was left out or not, and how many
# are older than it.
PAGE_END = """
WITH page AS (
    SELECT id FROM changes WHERE id < :older_than ORDER BY id DESC LIMIT :newest
)
SELECT min(id), (SELECT count(*) FROM changes WHERE id < (SELECT min(id) FROM page))
FROM page"""
# What find shows on each side of a match.
SNIPPET_CHARS = 60
NOT_FINDABLE = (
    "find takes the text to look for, as text (a str) of one character or more."
)
STOPPED = "stopped"
# As fold.js words what a run left going when it ended other than stopped.
CUT_OFF = "cut off"
ENDED_BADLY = ("failed", STOPPED, CUT_OFF, "exhausted")
# The REPL's frame atop a raised traceback, plain or an exception group's, with its source
# line and any carets: the engine's, not the cell's, so the fold drops it from a cell's
# output, as the chat does (EXEC_FRAME in tree.js); a step the historian printed carried it.
ENGINE_FRAME = re.compile(
    r"^((?:Traceback|  \+ Exception Group Traceback) \(most recent call last\):\n)"
    r'((?:  \| )?)  File "[^"\n]*deep_reasoner[/\\][^"\n]*", line \d+, in \S+\n'
    r"\2    \S.*\n(?:\2    [ ~^]*\^[ ~^]*\n)?",
    re.MULTILINE,
)
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
    """A change the library tool made, a page the page tool opened, or a worked example
    left out, as a step of the agent it was for."""
    step = {"kind": kind} | {key: event.get(key) for key in fields}
    add_step(tree, event.get("node"), step)


def add_change(tree: Tree, event: Event) -> None:
    """A change the library tool made, its before and after marked as the tool gives
    them (marked())."""
    texts = {key: marked(event.get(key)) for key in ("before", "after")}
    add_logged("change", CHANGE_FIELDS, tree, event | texts)


def end_cell(tree: Tree, event: Event) -> None:
    cell = tree.cells.get(event.get("cell"))
    if cell is not None:
        cell |= {key: event.get(key) for key in ("code", "output", "status")}
        if isinstance(cell["output"], str):
            cell["output"] = ENGINE_FRAME.sub(r"\1", cell["output"])


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
    """What the run left going when it ended was stopped with it, when it was stopped,
    else cut off, as when the app stopped under it; a sub-agent timed to the end."""
    status = STOPPED if event.get("outcome") == STOPPED else CUT_OFF
    for row in tree.rows:
        if row["status"] == "running":
            row |= {"status": status, "seconds": event["t"] - row["started"]}
    for cell in tree.cells.values():
        if cell["status"] == "running":
            cell["status"] = status


FOLDS = {
    "turn.start": start_turn,
    "agent.start": start_agent,
    "think": think,
    "cell.start": start_cell,
    "cell.end": end_cell,
    "library.change": add_change,
    "page.open": partial(add_logged, "page", PAGE_FIELDS),
    LEFT_OUT: partial(add_logged, "left_out", LEFT_OUT_FIELDS),
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


def cut(
    value: Any,
    limit: int | None = MAX_VALUE_CHARS,
    total: float = MAX_TOTAL_CHARS,
) -> Any:
    """*value*, copied, with each text past *limit* characters (unless None) cut there,
    saying how much is left out, each list or dict past MAX_VALUE_DEPTH levels marked in
    its place, and, once *total* characters are copied, the rest of each list or dict
    left out, saying how much: aliases in a shared file's YAML can make millions of
    short texts."""
    left = total

    def copied(each: Any, depth: int) -> Any:
        nonlocal left
        left -= 1
        if isinstance(each, str):
            text = (
                each if limit is None or len(each) <= limit else cut_text(each, limit)
            )
            left -= len(text)
            return text
        if not isinstance(each, dict | list):
            return each
        if depth == MAX_VALUE_DEPTH:
            return NESTED_TOO_DEEP
        items = list(each.items() if isinstance(each, dict) else enumerate(each))
        kept = []
        for key, item in items:
            if left <= 0:
                break
            left -= len(key) if isinstance(key, str) else 0
            kept.append((key, copied(item, depth + 1)))
        more = len(items) - len(kept)
        if isinstance(each, list):
            return [item for _, item in kept] + ([MORE.format(more)] if more else [])
        copy = dict(kept)
        if more:
            # Keyed by the mark itself, lengthened past any key the dict holds.
            mark = MORE.format(more)
            while mark in copy:
                mark = f"…{mark}"
            copy[mark] = None
        return copy

    return copied(value, 0)


def error_of(output: Any) -> str | None:
    """What a failed cell raised, as its last traceback's exception lines, the engine's
    frame atop it dropped; else, as in a record from someone else, its output's last line."""
    if not isinstance(output, str):
        return None
    lines = output.strip().splitlines()
    return raised_error(output, raised=True) or (lines[-1] if lines else None)


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
        """The thoughts, cells, Library changes, pages and worked examples left out of
        message *turn*'s top agent, or of sub-agent *node*."""
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
    is when it has none, holds a real tag, so wasn't typed, or doesn't read as a
    decomposition, which the store refuses."""
    if "<repl>" in text or not any(map(text.__contains__, REAL_TAGS)):
        return text
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
    if not changed:
        return text
    # Written in place where that writes the same, as for a block of text, so a copy of
    # one the tool gave keeps its layout and comments.
    in_place = LONE_PLACEHOLDER.sub(r"\1<\2repl>\3", text)
    return (
        in_place if yaml.safe_load(in_place) == decomposition else dump(decomposition)
    )


def marked(value: Any) -> Any:
    """*value*, as the tool gives it: each text in it, at any depth, keys included, with
    each repl tag marked (MARKS)."""
    return each_text(value, partial(TO_MARK.sub, lambda found: MARKS[found.group()]))


def unmarked(value: Any) -> Any:
    """*value*, as given to the tool, with each mark read back: marked()'s inverse."""
    return each_text(
        value, partial(TO_UNMARK.sub, lambda found: UNMARKS[found.group()])
    )


def each_text(value: Any, change: Callable[[str], str]) -> Any:
    """*value* with each text in it, at any depth, keys included, made *change*(text)."""
    if type(value) is str:
        return change(value)
    if type(value) is dict:
        return {
            each_text(key, change): each_text(item, change)
            for key, item in value.items()
        }
    if type(value) in (list, tuple):
        return [each_text(item, change) for item in value]
    return value


def tag_written(line: str) -> str:
    """*line*, written as the real tag when it is a placeholder tag alone, its indent kept."""
    tag = line.strip()
    return line.replace(tag, REAL_TAGS[tag]) if tag in REAL_TAGS else line


def written_out(value: Any, what: str) -> str:
    """*value*, plain (plain()), as YAML text, each list or dict it shares written out
    where it is; a LibraryError, saying *what* it is, for one that holds itself, nests
    past MAX_NESTING lists and dicts, or is too long written out."""
    size, depth = expanded_shape(value)
    if depth == math.inf:
        raise LibraryError(
            f"{what} holds itself, so it never ends: give each list or dict once."
        )
    if depth > MAX_NESTING:
        raise LibraryError(
            f"{what} nests more than {MAX_NESTING} lists and dicts deep."
        )
    if size > MAX_YAML_CHARS:
        raise LibraryError(
            f"{what} may be at most {MAX_YAML_CHARS:,} characters; written out, this"
            " holds more."
        )
    return dump(value)


def from_turns(name: Any, turns: Any) -> str:
    """The worked example *name* written as YAML from its *turns*, as write_decomposition
    takes them, each as the engine's message, the tags written in; a LibraryError for a
    turn whose text holds a tag."""
    if type(name) is not str or type(turns) is not list:
        raise LibraryError(NOT_TURNS)
    name = unmarked(name)
    messages = [unmarked(turn_message(turn)) for turn in turns]
    # Each a dict of texts, as turn_message() checked.
    for number, turn in enumerate(unmarked(turns), start=1):
        for field_name, text in turn.items():
            if found := TAG.search(text):
                tag = found.group()
                raise LibraryError(
                    TAG_IN_TURN.render(
                        number=number,
                        field=field_name,
                        tag=tag,
                        placeholder=f"[{tag[1:-1]}]",
                    )
                )
    return dump({"name": name, "messages": messages})


def turn_message(turn: Any) -> dict[str, str]:
    """A worked example's *turn*, {user}, {think, code} or {observation}, as the engine's
    message: a step's code as typed indented in a cell is written from its first column."""
    if type(turn) is not dict or any(
        type(each) is not str for each in [*turn, *turn.values()]
    ):
        raise LibraryError(NOT_TURNS)
    if set(turn) == {"user"}:
        return {"role": "user", "content": turn["user"]}
    if set(turn) == {"observation"}:
        printed = turn["observation"].strip("\n")
        return {"role": "user", "content": f"<observation>\n{printed}\n</observation>"}
    if "code" not in turn or not set(turn) <= set(STEP):
        raise LibraryError(NOT_TURNS)
    code = textwrap.dedent(turn["code"]).strip("\n")
    think = f"<think>{turn['think'].strip()}</think>\n" if "think" in turn else ""
    return {"role": "assistant", "content": f"{think}<repl>\n{code}\n</repl>"}


def yaml_text(value: Any, typed: Callable[[str], str] = str) -> str:
    """*value*, YAML text or a dict of plain values, as YAML text, from its first line
    that isn't blank, its marks read back (unmarked()) once *typed* has written what was
    typed."""
    if type(value) is dict and plain(value):
        # Read back once checked, as a dict holding itself never ends.
        written_out(value, "The YAML")
        return typed(dump(unmarked(value)))
    # Exactly str: a subclass could compare unequal to "root" and still name it.
    if type(value) is not str:
        raise LibraryError(NOT_TEXT)
    return unmarked(typed(value.lstrip("\n")))


def check_whole(texts: Iterable[str]) -> None:
    """Raise if any of *texts* holds the mark of a text a brief list cut short."""
    if any(CUT_MARK.search(text) for text in texts):
        raise LibraryError(CUT_SHORT)


def check_writable(namespace: str, text: str, key_names: list[str]) -> Any:
    """Raise unless the tool may write the YAML *text* for the namespace named
    *namespace*; *text* as data."""
    if type(namespace) is not str:
        raise LibraryError(NOT_TEXT)
    if namespace == ROOT:
        raise LibraryError(ROOT_LOCKED)
    check_size(text, "The YAML")
    check_whole([text])
    # Parsed too, as an escape such as "\x73k-" spells a key the raw text doesn't hold.
    data = loaded(text)
    check_no_key([namespace, text, *texts_in(data)], key_names)
    return data


def code_problems(decomposition: dict[str, Any]) -> list[str]:
    """Each assistant turn of *decomposition*, as data the store took, whose code, the
    <repl> block the engine runs, isn't valid Python, or leaves the {name} of a variable
    its code or an earlier turn's sets in a string without f."""
    problems, names = [], set()
    for number, turn in enumerate(decomposition["messages"], start=1):
        if turn["role"] != "assistant":
            continue
        what = f"Turn {number} (assistant)'s code"
        source = running_code(turn["content"])
        try:
            tree = parsed_python(source, what)
        except LibraryError as error:
            problems.append(str(error))
            continue
        names |= set_names(tree)
        problems += [
            UNFILLED.render(what=what, name=name)
            for name in dict.fromkeys(fields_without_f(source, tree))
            if name in names
        ]
    return problems


def set_names(tree: ast.AST) -> set[str]:
    """The names the code *tree* sets: by assignment, a loop or a function's argument."""
    stored = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    return stored | {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}


def fields_without_f(source: str, tree: ast.AST) -> Iterator[str]:
    """The name in each {name} field of each string in *source*, parsed as *tree*,
    written without f, but those .format() fills in."""
    formatted = [
        node.func.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in FORMAT_METHODS
    ]
    spans = [
        ((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset))
        for node in formatted
    ]
    # A parsed source tokenizes: f-strings come as tokens of their own, never STRING.
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.STRING:
            continue
        if any(start <= token.start and token.end <= end for start, end in spans):
            continue
        value = ast.literal_eval(token.string)
        if isinstance(value, str):
            yield from FIELD.findall(value)


def plain(value: Any) -> bool:
    """Whether *value* is text, a number, True, False or None, or a list or dict of them
    keyed by text; one it shares is read once, as written_out() measures how deep it
    nests and whether it holds itself."""
    stack, seen = [value], set()
    while stack:
        each = stack.pop()
        if type(each) in PLAIN or id(each) in seen:
            continue
        if type(each) not in (list, dict):
            return False
        seen.add(id(each))
        if type(each) is dict and any(type(key) is not str for key in each):
            return False
        stack.extend(each.values() if type(each) is dict else each)
    return True


def secret_like(name: Any) -> bool:
    """Whether *name*, a var's or setting's, reads as a secret's (SECRET_PART,
    SECRET_ENDING)."""
    if type(name) is not str:
        return False
    parts = WORD_PART.findall(name)
    return any(map(SECRET_PART.fullmatch, parts)) or bool(SECRET_ENDING.search(name))


def key_shaped(text: str) -> bool:
    """Whether *text* holds a run of a key's characters that reads as random."""
    for run in KEY_RUN.findall(text):
        letters_and_digits = re.sub(r"[^A-Za-z0-9]", "", run)
        switches = len(ONE_CASE_OR_DIGITS.findall(letters_and_digits)) - 1
        share = switches / max(len(letters_and_digits), KEY_LENGTH)
        if switches >= MIN_SWITCHES or share >= MIN_SWITCH_SHARE:
            return True
    return False


def secret_shaped(name: Any, value: Any) -> bool:
    """Whether *value*, held under *name*, is text or bytes holding a key, and not the
    name of the environment variable that holds one, as a name ending in env gives."""
    if type(value) is bytes:
        value = value.decode("latin-1")
    if type(value) is not str or not key_shaped(value):
        return False
    parts = WORD_PART.findall(name) if type(name) is str else []
    names_a_variable = [part.lower() for part in parts[-1:]] == [NAMES_A_VARIABLE]
    return not (names_a_variable and VARIABLE_NAME.fullmatch(value))


def stored_at(stored: Any, path: tuple[Any, ...]) -> Any:
    """What *stored* holds at *path*, its keys and indexes in turn; ABSENT for none."""
    for step in path:
        try:
            stored = stored[step]
        except (KeyError, IndexError, TypeError):
            return ABSENT
    return stored


def check_no_secrets(field: str, values: dict[str, Any], stored: Any) -> None:
    """Raise if *values*, a namespace's vars or a tool's settings, holds anywhere text
    shaped like a key under a name like a secret's, its own or one it is held under,
    unless *stored* holds it in the same place: the user's own stays. The refusal names,
    for each, the nearest name over it like a secret's."""
    found = set()
    stack = [((name,), name, value, None) for name, value in values.items()]
    while stack:
        path, name, value, under = stack.pop()
        under = name if secret_like(name) else under
        if type(value) is dict:
            stack.extend(
                ((*path, key), key, each, under) for key, each in value.items()
            )
        elif type(value) is list:
            stack.extend(
                ((*path, at), name, each, under) for at, each in enumerate(value)
            )
        elif under and secret_shaped(name, value) and stored_at(stored, path) != value:
            found.add(str(under))
    if found:
        raise LibraryError(secret_refusal(field, sorted(found)))


def secret_refusal(field: str, names: list[str]) -> str:
    """Why *field* may not set *names*, named like secrets, to secrets, keeping the first
    in a variable named as its words in capitals, less a last env, if that names one."""
    words = WORD_PART.findall(names[0])
    if words and words[-1].lower() == NAMES_A_VARIABLE:
        words.pop()
    variable = "_".join(words).upper()
    if not VARIABLE_NAME.fullmatch(variable):
        variable = ANY_VARIABLE
    return SECRET_LIKE.render(field=field, names=names, variable=variable)


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
    if problem := tool_name_problem(name):
        raise LibraryError(problem)
    check_tool_block(name, {"factory": factory, "factory_from": tool_file(name)})
    check_size(source, "The source")
    check_size(written_out(settings, "The settings as YAML"), "The settings as YAML")
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
            raise locked(key, name, new)
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


def locked(key: str, name: str, adding: bool) -> LibraryError:
    """The refusal to set or change namespace *name*'s *key*, repl or reasoner, as a new
    one when *adding*."""
    where = (
        f"Namespaces → + Add namespace, set in {name}'s YAML"
        if adding
        else f"Namespaces → {name} → Edit YAML"
    )
    return LibraryError(
        f"{key} may not be set or changed: {LOCKED_FIELDS[key]}, in the Library panel:"
        f" {where}."
    )


def row_fields(read: dict[str, Any], name: str) -> dict[str, Any]:
    """The fields namespace *name*'s row in the Library *read* (read_library()) sets,
    whether or not the engine reads it."""
    return fields_of(read["namespace_yaml"].get(name) or "")


def check_unlocked(read: dict[str, Any], name: str, text: str, adding: bool) -> None:
    """Raise if the YAML *text* sets namespace *name*'s repl or reasoner otherwise than
    its row in the Library *read* does: before the engine reads it, so the lock is the
    refusal, not a field a sandbox's settings lack."""
    given, own = fields_of(text), row_fields(read, name)
    for key in LOCKED_FIELDS:
        if key in given and given[key] != own.get(key):
            raise locked(key, name, adding)


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
    """A namespace as show() lists it, before its texts are cut short: what it sets
    itself, the tools its agents get, and each of its own decompositions with its
    problem count."""
    return {
        "name": row["name"],
        "depth": row["depth"],
        "sets": fields_of(row["yaml"]),
        "tools": list(registry.resolve(row["name"]).tools),
        "decompositions": [
            {"name": each["name"], "problems": len(each["problems"])}
            for each in row["decompositions"]
        ],
    }


def in_full(
    row: dict[str, Any],
    read: dict[str, Any],
    run: V2Config,
    registry: NamespaceRegistry,
) -> dict[str, Any]:
    """Namespace *row* (namespace_rows()) of the Library *read* as show(name) gives it;
    *run* and *registry* as run_config_and_registry(config_of(*read*)) makes them."""
    name = row["name"]
    held = registry.resolve(name).sources.decompositions
    own, left_out = shown_here(read, row, held)
    stored = {each.name: each for each in registry.resolve(name).decompositions}
    as_run = without_impossible_hand_offs(config_of(read), run, registry)
    decompositions = []
    for example in run_config_and_registry(as_run)[1].resolve(name).decompositions:
        entry = {"name": example.name, "from": held[example.name]}
        entry |= own.get(example.name, {})
        if example != stored[example.name]:
            entry["rewritten"] = True
        decompositions.append(entry)
    left = read["namespaces_left"]
    return {
        "name": name,
        "depth": row["depth"],
        "yaml": left_out_note(left[name]) if name in left else row["yaml"],
        "fields": row["fields"],
        "decompositions": decompositions,
        "decompositions_left_out": left_out,
    }


def shown_here(
    read: dict[str, Any], row: dict[str, Any], held: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Namespace *row*'s own decompositions: the YAML and problems of each its agents
    are shown, by the name the engine reads, as *held* says where each comes from; and
    the rest, which none is, as a shared file's unreadable row or a root left out."""
    name = row["name"]
    rows = [each for each in read["decompositions"] if each["namespace"] == name]
    own, left_out = {}, []
    for each, entry in zip(rows, row["decompositions"], strict=True):
        found = each["decomposition"]
        if found is not None and held.get(found.name) == name:
            own[found.name] = {"yaml": entry["yaml"], "problems": entry["problems"]}
        else:
            left_out.append(entry)
    return own, left_out


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
    tool's before and after as write_tool takes them. A text left out, a note saying
    why, is said so, and changes no field."""
    change = {key: event.get(key) for key in CHANGE_FIELDS}
    if noted := left_out_texts(change):
        change["unreadable"] = CANT_WRITE_BACK.format(" and ".join(noted))
    elif change["what"] == "namespace":
        before, after = change["before"], change["after"]
        change["changed"] = field_changes(before or "", after)[0]
    if change["what"] == "tool":
        for key in ("before", "after"):
            change[key] = tool_fields(change["name"], change[key])
    return change


def left_out_texts(change: Event) -> list[str]:
    """Which of a *change*'s texts, before and after, are notes saying why its row was
    left out."""
    return [key for key in ("before", "after") if is_left_out_note(change[key])]


def change_in_brief(event: Event) -> dict[str, Any]:
    """A library.change *event* in a line: what it did, to what, and the fields it
    changed in a namespace."""
    change = {key: event.get(key) for key in CHANGE_FIELDS}
    done = "added" if change["before"] is None else "changed"
    brief = {"done": done, "what": change["what"]}
    if change["what"] != "tool":
        brief["namespace"] = change["namespace"]
    brief["name"] = change["name"]
    if change["what"] == "namespace":
        brief["fields"] = list(
            field_changes(change["before"] or "", change["after"])[0]
        )
    return brief


def chat_changes(library: str, chat: str, turn: Any) -> list[dict[str, Any]]:
    """The Library changes of *chat*'s message *turn*, whole, or of every message of
    its, this one's so far included, in brief; each with its turn and message."""
    with app_file(library, "ro") as connection:
        turns = chat_turns(connection, chat)
    if turn is not None and (type(turn) is not int or not 1 <= turn <= len(turns)):
        raise ValueError(
            f"There is no message {turn!r}: this chat's messages are numbered 1 to"
            f" {len(turns)}, this one last."
        )
    whole = turn is not None
    found, limit = [], MAX_VALUE_CHARS if whole else BRIEF_CHARS
    for number, each in enumerate(turns, start=1):
        if turn not in (None, number):
            continue
        events = each["events"]
        at = {"turn": number, "message": cut(events[0]["prompt"], limit=limit)}
        found += [
            at | (as_written(event) if whole else change_in_brief(event))
            for event in events
            if event["kind"] == "library.change"
        ]
    return found


def recorded_changes(library: str, chat: str, older_than: Any) -> list[Event]:
    """The Library's newest MAX_FOUND changes, whoever made them, older than the one of
    id *older_than* when given, each in a line, then how many more; *chat* is this
    run's, whose changes say so."""
    check_change_id(older_than, "older_than")
    page = {"newest": MAX_FOUND, "older_than": older_than or sys.maxsize}
    with app_file(library, "ro") as connection, reading(connection):
        found = read_changes(connection, MAX_FOUND, older_than)
        turns = turns_of(connection, chat, found)
        [(last, more)] = connection.execute(PAGE_END, page)
    lines = [
        cut(change_head(each, chat, turns) | recorded_in_brief(each), BRIEF_CHARS)
        for each in found
    ]
    if more:
        note = f"{more:,} older not listed: changes(older_than={last}) lists them."
        lines.append({"more": more, "note": note})
    return lines


def recorded_change(library: str, chat: str, id: Any) -> Event:
    """The Library's change of id *id*, whole, as changes(turn) gives this chat's."""
    check_change_id(id, "id")
    with app_file(library, "ro") as connection, reading(connection):
        found = read_changes(connection, 1, id + 1)
        turns = turns_of(connection, chat, found)
    if not found or found[0]["id"] != id:
        raise ValueError(f"There is no change {id}: changes() lists them by id.")
    return change_head(found[0], chat, turns) | as_written(found[0])


def check_change_id(value: Any, name: str) -> None:
    if value is not None and not (type(value) is int and 0 < value < sys.maxsize):
        raise ValueError(f"{name} should be a change's id, as changes() lists them.")


def turns_of(
    connection: sqlite3.Connection, chat: str, changes: list[Event]
) -> dict[str, int]:
    """The turn of *chat*'s, by its run, that made each of *changes* made there,
    numbered as chat_turns() numbers them, from each run's first event alone."""
    if not any(change["chat"] == chat for change in changes):
        return {}
    runs = chat_runs(connection, chat)
    turns = [run for run in runs if started(first_loaded(connection, run, FORWARD))]
    return {run: number for number, run in enumerate(turns, start=1)}


def change_head(change: Event, chat: str, turns: dict[str, int]) -> Event:
    """A recorded *change*'s id, when it was made, by whom (this chat's, *chat*, with
    the turn, by *turns*, of the run that made it), and what it did."""
    head = {"id": change["id"], "when": day_of(change["made"])}
    head["by"] = made_by(change, chat)
    if change["chat"] == chat and change["run"] in turns:
        head["turn"] = turns[change["run"]]
    return head | {"done": done_of(change)}


def done_of(change: Event) -> str:
    if change["before"] is None:
        return "added"
    return "deleted" if change["after"] is None else "changed"


def day_of(made: float) -> str:
    """The day a change was *made*: today, yesterday, else its date."""
    day, today = datetime.fromtimestamp(made).date(), date.today()
    named = {today: "today", today - timedelta(days=1): "yesterday"}
    return named.get(day, day.isoformat())


def made_by(change: Event, chat: str) -> str:
    """Who made a recorded *change*, in words: the panel, an import, this chat
    (*chat*) or another, by its title."""
    if change["by"] == "panel":
        return "the Library panel"
    if change["by"] == "import":
        return f"an import of {change['label']}"
    if change["chat"] == chat:
        return "this chat"
    title = change["label"]
    return "a chat since deleted" if title is None else f'the chat "{title}"'


def recorded_in_brief(change: Event) -> Event:
    """A recorded *change*'s row, in a line: what, where and its name, and a
    namespace's fields it changed."""
    brief = {"what": change["what"]}
    if change["namespace"] is not None:
        brief["namespace"] = change["namespace"]
    brief["name"] = change["name"]
    if change["what"] == "namespace" and not left_out_texts(change):
        changed = field_changes(change["before"] or "", change["after"] or "")[0]
        brief["fields"] = list(changed)
    return brief


def check_builds(
    tools: dict[str, tuple[dict[str, Any], str]], tools_dir: Path, done: str
) -> None:
    """Raise unless each of *tools*, by name its block and Python source, builds as a run
    of this run's config, in its folder beside *tools_dir*, would build it, in this
    worker's environment. *done* says what wasn't done when one doesn't."""
    if not tools:
        return
    config = tool_check.config_of(tools_dir.parent / SPEC)
    try:
        refuse_unbuilt(tools, config, os.environ, tools_dir / CHECKS, done)
    except LibraryError as error:
        notes = module_notes(str(error))
        raise LibraryError("\n".join([str(error), *notes])) from None


def module_notes(refusal: str) -> list[str]:
    """What the user does about each module the build's *refusal* says a tool couldn't
    import: add its package to the app's Python; or, a package this worker finds, which
    the build couldn't, as it is in the chat's folder, put it in the tool. A module an
    installed package lacks needs neither."""
    notes = []
    for module in dict.fromkeys(MISSING_MODULE.findall(refusal)):
        package = module.partition(".")[0]
        try:
            found = find_spec(package) is not None
        except (ImportError, ValueError):
            # As for one made in code: what holds it is unknown, so say nothing more.
            continue
        if not found:
            notes.append(NOT_INSTALLED.render(module=package))
        elif module == package:
            notes.append(IN_FOLDER_ONLY.render(module=module))
    return notes


def giving_marks(method: Callable[..., Any]) -> Callable[..., Any]:
    """The library tool's *method*, its result given with each repl tag marked."""

    @wraps(method)
    def given(*args: Any, **kwargs: Any) -> Any:
        return marked(method(*args, **kwargs))

    return given


def build_library(client: Any, params: dict[str, Any]) -> Func:
    """The library tool, over the app file *library* its block in the run's config names,
    for a run of a chat in *namespace*, whose worker holds keys in *key_names*; it builds
    the tools it checks in the run's *tools_dir*."""
    library, chat_namespace = params["library"], params["namespace"]
    key_names, tools_dir = params["key_names"], Path(params["tools_dir"])
    editors = params.get("editors", [])
    pick = PICK_AN_EDITOR.render(editors=editors)
    by = by_chat(params["chat"], params["run"])

    def show(name: str | None = None, whole: bool = False) -> Any:
        """Every namespace in brief, or namespace *name* in full: its fields as the panel
        shows them, and the decompositions its agents are shown, as a run resolves them,
        each text cut past SHOWN_CHARS; or, *whole*, its own YAML and decompositions
        uncut, without the fields, which repeat them. Not the panel's Delete refusals:
        an agent that can't delete took them as reasons not to edit."""
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
            return cut([in_brief(row, registry) for row in rows], limit=BRIEF_CHARS)
        found = in_full(
            named(rows, name, "namespace", "show()"), read, config, registry
        )
        if not whole:
            return cut(found, limit=SHOWN_CHARS)
        # Its own YAML and worked examples, not the fields, which repeat what they hold.
        stored = {key: value for key, value in found.items() if key != "fields"}
        return cut(stored, limit=None, total=math.inf)

    def tools(name: str | None = None) -> Any:
        """Every tool in brief, or tool *name* whole, with its source."""
        with app_file(library, "ro") as connection, reading(connection):
            rows = tool_entries(connection)
        if name is None:
            return cut(rows, limit=BRIEF_CHARS)
        return named(rows, name, "tool", "tools()")

    def changes(
        turn: int | None = None, id: int | None = None, older_than: int | None = None
    ) -> Any:
        """Every change to the Library, newest first, each in a line; or the one of id
        *id*, whole; or this chat's in its message *turn*, whole, as they were logged."""
        if id is not None:
            return recorded_change(library, params["chat"], id)
        if turn is not None:
            return chat_changes(library, params["chat"], turn)
        return recorded_changes(library, params["chat"], older_than)

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
        read: dict[str, Any],
    ) -> None:
        """Raise unless the tool may write *namespace*, as the Library stands (*read*),
        building the custom tools it grants *afresh* if any."""
        config, registry = run_config_and_registry(config_of(read))
        check_grant(registry)
        check_locks(config, registry, namespace)
        stored = row_fields(read, namespace.name).get("vars")
        check_no_secrets("vars", namespace.vars, stored)
        if afresh:
            check_local(registry)

    def may_write_tool(
        connection: sqlite3.Connection, name: str, settings: dict[str, Any]
    ) -> None:
        config, registry = config_and_registry(connection)
        check_grant(registry)
        check_local(registry)
        check_tool_locks(config, registry, name)
        check_no_secrets("settings", settings, config.tools.get(name, {}))

    def grants_built(
        name: str, text: str, adding: bool, merge: bool = False
    ) -> Callable[[sqlite3.Connection, NamespaceConfig], None]:
        """The write's *allowed* for namespace *name* as the YAML *text* makes it, once
        the custom tools it grants afresh are built (built_grants())."""
        build = partial(check_builds, tools_dir=tools_dir, done=GRANTED)

        def first(read: dict[str, Any]) -> None:
            check_grant(run_config_and_registry(config_of(read))[1])
            check_unlocked(read, name, text, adding)

        with app_file(library, "ro") as connection:
            built = built_grants(
                connection, name, text, build, adding, merge, locks=locks, first=first
            )
        return partial(same_grants, built, locks)

    def write_decomposition(
        namespace: str, yaml: Any = None, turns: Any = None, name: Any = None
    ) -> dict[str, Any]:
        """Add to *namespace* the worked example named *yaml* (or *name*), written from
        its *turns*, or else the decomposition *yaml*; or replace the one of its name.
        The documented calls are write_decomposition(namespace, name, turns) and
        write_decomposition(namespace, yaml)."""
        if name is not None and (turns is None or yaml is not None):
            raise LibraryError(TWO_FORMS)
        if turns is not None:
            text = from_turns(yaml if name is None else name, turns)
        else:
            text = yaml_text(yaml, typed=with_real_tags)
        data = check_writable(namespace, text, key_names)

        def write() -> tuple[dict[str, Any], Event]:
            stored, before = written(
                library,
                add_decomposition,
                namespace,
                text,
                replace=True,
                allowed=may_write,
                by=by,
            )
            at = {
                "what": "decomposition",
                "namespace": namespace,
                "name": stored["name"],
            }
            if before == text:
                return stored, {"done": NOTHING_CHANGED, **at}
            done = "added" if before is None else "replaced"
            return stored, logged(done, before, text, **at)

        stored, change = stop_proof(write)
        broken = code_problems(data)
        return {**change, "problems": stored["problems"]} | (
            {"code_problems": broken} if broken else {}
        )

    def add(name: str, yaml: Any) -> dict[str, Any]:
        """Add namespace *name* with the fields *yaml* sets."""
        text = yaml_text(yaml)
        check_writable(name, text, key_names)
        allowed = grants_built(name, text, adding=True)

        def write() -> Event:
            stored = written(library, add_namespace, name, text, allowed=allowed, by=by)
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
                library,
                change_namespace,
                name,
                text,
                allowed=allowed,
                merge=merge,
                by=by,
            )
            done = None if stored == before else logged("changed", before, stored, **at)
            return before, stored, done

        before, stored, done = stop_proof(write)
        if done is None:
            return {"done": NOTHING_CHANGED, **at}
        changed, kept = field_changes(before, stored)
        # A text given, yet as it was: a .replace(...) that matched nothing, say.
        given = fields_of(text).items() if merge else ()
        same = [
            key for key, value in given if type(value) is str and key not in changed
        ]
        done |= {"changed": cut(changed), "kept": [k for k in kept if k not in same]}
        if same:
            done["nothing changed"] = f"{', '.join(same)} {ALREADY_HELD}"
        return done

    def write_tool(
        name: str, source: str, factory: str, settings: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Add tool *name*, built by the function *factory* in the Python *source* with
        *settings* as its params, or replace the one of its name."""
        source = unmarked(source)
        check_tool(name, source, factory, settings, key_names)
        # Read back once checked, as settings holding themselves never end.
        settings = unmarked(settings)
        # The grant and the locks first, before the tool's code runs; the build holds no
        # write lock, so they are checked again under it.
        with app_file(library, "ro") as connection:
            may_write_tool(connection, name, settings or {})
        block = {
            "factory": factory,
            "factory_from": tool_file(name),
            **(settings or {}),
        }
        check_builds({name: (block, source)}, tools_dir, WRITTEN)

        def write() -> Event:
            allowed = partial(may_write_tool, name=name, settings=settings or {})
            before = written(
                library, store_tool, name, block, source, allowed=allowed, by=by
            )
            after = tool_text(name, dump(block), source)
            if before == after:
                return {"done": NOTHING_CHANGED, "what": "tool", "name": name}
            done = "added" if before is None else "replaced"
            return logged(done, before, after, what="tool", name=name)

        return stop_proof(write)

    tool = SimpleNamespace(
        show=giving_marks(show),
        tools=giving_marks(tools),
        changes=giving_marks(changes),
        write_decomposition=giving_marks(write_decomposition),
        add_namespace=giving_marks(add),
        change_namespace=giving_marks(change),
        write_tool=giving_marks(write_tool),
    )
    description = LIBRARY_DESCRIPTION
    if chat_namespace not in editors:
        description += READ_ONLY_HERE.render(namespace=chat_namespace, pick=pick)
    return Func(tool, description=description)
