"""The app file: the Library (every namespace, decomposition and custom tool, and the run
settings) in one SQLite file, which also holds the chats, their runs' events and the app's
own settings, so a copy of it carries the whole app but for the keys and the runs' working
folders.

Each Library row holds the engine's own YAML, so the file turns back into a deep_reasoner
config at run start. chats.py, runs.py and settings.py keep the chats, the runs' events and
the app's settings; this module lays the whole file out.
"""

import ast
import builtins
import inspect
import json
import keyword
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from functools import cache, lru_cache, partial
from pathlib import Path

import deep_reasoner.llm
import jinja2
import structlog
import yaml
from deep_reasoner.namespaces import (
    CONFIG_SOURCE,
    ROOT,
    NamespaceConfig,
    NamespaceSpawnError,
    check_spawn,
)
from deep_reasoner.primitives import RESERVED_NAMES
from deep_reasoner.prompt_config import Decomposition
from deep_reasoner.tools import TOOL_BUILDERS
from deep_reasoner.tools.base import DEFAULT_CLAUDE_DESCRIPTION, DEFAULT_RAG_DESCRIPTION
from deep_reasoner.tools.kg_main import KnowledgeGraph
from deep_reasoner.tools.safe_url import DEFAULT_SAFE_URL_DESCRIPTION
from deep_reasoner.v2.agent import DeepReasoner
from deep_reasoner.v2.claude_reasoner import ClaudeDeepReasoner
from deep_reasoner.v2.cli import (
    _DEFAULT_LLM_DESCRIPTION,
    V2Config,
    build_llm_tool,
    build_namespace_registry,
    declared_tool_names,
)
from deep_reasoner.v2.messages import NoCodeBlock, _has_jinja, code
from jinja2 import (
    DictLoader,
    Environment,
    TemplateError,
    TemplateSyntaxError,
    meta,
    nodes,
)
from jinja2.sandbox import (
    UNSAFE_ASYNC_GENERATOR_ATTRIBUTES,
    UNSAFE_COROUTINE_ATTRIBUTES,
    UNSAFE_FUNCTION_ATTRIBUTES,
    UNSAFE_GENERATOR_ATTRIBUTES,
    UNSAFE_METHOD_ATTRIBUTES,
    SandboxedEnvironment,
)
from pydantic import ValidationError

log = structlog.get_logger()

SCHEMA_VERSION = 4
# Version 1 held the Library alone, version 2 added the chats and the app's settings, and
# version 3 the runs; a file of any opens and gains the tables of the versions after it.
LIBRARY_ONLY = 1
BEFORE_EVENTS = 2
BEFORE_PAGES = 3
LIBRARY_SCHEMA = (
    "CREATE TABLE settings (key TEXT PRIMARY KEY, yaml TEXT NOT NULL)",
    "CREATE TABLE namespaces (name TEXT PRIMARY KEY, yaml TEXT NOT NULL)",
    (
        "CREATE TABLE decompositions (namespace TEXT NOT NULL, name TEXT NOT NULL,"
        " yaml TEXT NOT NULL, PRIMARY KEY (namespace, name))"
    ),
    "CREATE TABLE tools (name TEXT PRIMARY KEY, yaml TEXT NOT NULL, source TEXT)",
)
LIBRARY_TABLES = {"settings", "namespaces", "decompositions", "tools"}
APP_TABLES = {"chats", "app_settings"}
# A chat's choices are null until picked or run, and a null folder is its scratch folder.
# app_settings is the app's, as JSON: the Library's own settings table is the run settings.
APP_SCHEMA = (
    (
        "CREATE TABLE chats (id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL NOT NULL,"
        " namespace TEXT, provider TEXT, model TEXT, folder TEXT)"
    ),
    "CREATE TABLE app_settings (key TEXT PRIMARY KEY, json TEXT NOT NULL)",
)
RUN_TABLES = {"runs", "events"}
# A run's id is when it began, in UTC, to the microsecond, so ids sort as their runs began.
RUN_ID = r"^\d{8}-\d{6}-\d{6}$"
RUN_ID_FORMAT = "%Y%m%d-%H%M%S-%f"
# Each run's chat, and its events as the JSON the page folds, numbered from 1 (seq).
RUN_SCHEMA = (
    "CREATE TABLE runs (id TEXT PRIMARY KEY, chat TEXT NOT NULL)",
    "CREATE INDEX runs_by_chat ON runs (chat)",
    (
        "CREATE TABLE events (run TEXT NOT NULL, seq INTEGER NOT NULL, data TEXT NOT NULL,"
        " PRIMARY KEY (run, seq))"
    ),
)
PAGE_TABLES = {"pages"}
# What the page tool saved: each page's chat, the run and agent (node) that made it, and
# its blocks as JSON. node has no type, so it keeps the record's int or text.
PAGE_SCHEMA = (
    (
        "CREATE TABLE pages (id TEXT PRIMARY KEY, chat TEXT NOT NULL, run TEXT NOT NULL,"
        " node, title TEXT NOT NULL, blocks TEXT NOT NULL, created REAL NOT NULL)"
    ),
    "CREATE INDEX pages_by_chat ON pages (chat)",
)
# The tables each version added, and how.
ADDED_IN = {
    LIBRARY_ONLY: (LIBRARY_TABLES, LIBRARY_SCHEMA),
    BEFORE_EVENTS: (APP_TABLES, APP_SCHEMA),
    BEFORE_PAGES: (RUN_TABLES, RUN_SCHEMA),
    SCHEMA_VERSION: (PAGE_TABLES, PAGE_SCHEMA),
}
# Upserts keep a row's place, so decompositions stay in the order they were added.
UPSERT_SETTING = "INSERT INTO settings VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET yaml = excluded.yaml"
UPSERT_NAMESPACE = "INSERT INTO namespaces VALUES (?, ?) ON CONFLICT (name) DO UPDATE SET yaml = excluded.yaml"
UPSERT_DECOMPOSITION = (
    "INSERT INTO decompositions VALUES (?, ?, ?)"
    " ON CONFLICT (namespace, name) DO UPDATE SET yaml = excluded.yaml"
)
UPSERT_TOOL = (
    "INSERT INTO tools VALUES (?, ?, ?)"
    " ON CONFLICT (name) DO UPDATE SET yaml = excluded.yaml, source = excluded.source"
)
# Granted at root, so every namespace below has it, as in the engine's example configs.
FRESH_ROOT = {"tools": ["llm"]}
# What app its agents are in, which every namespace inherits; a fresh Library's only.
ROOT_SUFFIX = """\
You are an agent of Deep Reasoning, an app on the user's own computer: agents like you \
answer each chat message in Python, run unsandboxed in the chat's working folder (files \
you write go there, never /mnt/data: give each one's path in code, never a link), \
handing parts to sub-agents; the ✻ line under each answer opens its tree of steps and \
code, and a page is never a file: hand every page ask, of any data, on as your worked \
examples do. Never install a package: name the missing one, which the user adds by \
restarting the app as `uv run --with PKG deep-reasoning`. What they know is the app's \
Library, saved with the chats in its file (by default ~/.deep-reasoning/library.sqlite) \
and edited by the user in the Library panel (top right): each namespace's system_suffix \
(root's and yours included: the user's own text, no secret), tools and worked examples, \
root's reaching every chat.
Compute each number in Python and print it, then answer from the variables, in that cell \
or a later one, as FinalAnswer(f"The mean of {n} values is {mean:.2f}."): never type a \
number you computed. When the user names a namespace or tool to use, pass on its answer, \
computing nothing beside it, and say if it couldn't be used. Asked if an earlier answer \
was right, check its record rather than work it out again. Combine sub-agents' results \
into one answer in prose or a Markdown table, never a raw dict or list, and no LaTeX. \
Only your FinalAnswer reaches whoever asked: put your results in it, and leave out your \
failed cells unless asked.
"""
# library writes the Library, so it is granted in a namespace of its own, which a chat
# picks when it wants edits, and which no namespace inherits from.
LIBRARIAN = "librarian"
LIBRARY_TOOL = "library"
LIBRARIAN_SUFFIX = """\
You are the librarian, the namespace the user picks to change this app's Library, \
which you do with the library tool. Your jobs: add namespaces and change their fields \
(system_suffix, vars, tools, spawn); write worked examples into them; write custom \
Python tools, fix them and grant them; and undo the changes made in this chat. Asked \
what you can do, say these jobs. Read before you write: library.show() lists the \
namespaces, library.tools() the tools. Write only what the user asked for: to change a \
text such as a system_suffix, start from its current text (show(name)) and keep every \
line they didn't ask to remove; leave spawn unset unless they ask to restrict it, and \
then keep historian and librarian in it, as root hands them questions about earlier \
messages and about the Library. Make team members siblings, not children of their \
planner, unless they should inherit its suffix and worked examples; say which you did. \
Answer by saying what changed, as the tool's result says, and anything you removed. \
Quote what the user asked to see from a value, or from what an earlier cell printed. A \
change applies from the user's next message.
To change a worked example already stored, take its yaml from library.show(name) and \
edit it with .replace(...) in your cell; never type it out again, as its own repl tags \
would end your cell and run its code instead. To undo a change, find it with \
library.changes() and write its before back; never write a before from memory.
Say when a worked example disagrees with its namespace's suffix. Only the user can \
change what the tool locks, or delete anything: say so plainly, and how, as the tool's \
refusals and description do.
"""
# It may hand questions about earlier messages to the historian, but never spawn itself:
# copying root's hand-off into its own namespace would nest it until max_depth. A
# default, not a lock: the user may widen it in the panel.
FRESH_LIBRARIAN = {
    "tools": [LIBRARY_TOOL],
    "spawn": ["historian"],
    "system_suffix": LIBRARIAN_SUFFIX,
}
# A specialist that root hands questions about earlier messages to.
HISTORIAN = "historian"
HISTORY_TOOL = "history"
HISTORIAN_SUFFIX = """\
You are the historian: you answer questions about this chat's earlier messages (how or \
why an answer was reached, what an agent did, what went wrong) from what was recorded. \
The question was handed to you, so never hand it on with subagent: read the record with \
history and quote the real code and outputs. Never redo the work or guess; answer in a \
later cell, from what you printed, and never repeat a read you already printed.
Only what a cell printed, or passed to FinalAnswer as an expression, shows what code \
gave: a value typed into FinalAnswer(...) by hand is the agent's claim, so say the record \
doesn't show it was computed.
The user sees each run as one tree and may call any agent in it a sub-agent, so what \
they point at may be the top agent's own cell, such as the one that delegated and then \
combined the sub-agents' results. Find an error with history.errors(turn), which reads \
every agent, the top one included, and anything else the user names with history.find. \
If it isn't where the user says, say where it is. Leave out failures of earlier historian \
runs (namespace historian), such as a refused hand-off, unless the user asks about them.
Name each agent by what it was asked, never by its node number, and leave out how you \
read the record (history.errors, failed: 0).
Make a page only when the user asks to see something and you found it; when you found \
nothing, say so in words and make no page. Fill a page's tables from the record in code, \
never by typing its values. Your answer is all that later messages will see of this run, \
so say what you found, in a sentence or two, even when you make a page. Never write the \
page's link, or that it is open or linked: the chat shows it under your answer.
"""
# A leaf: copying root's hand-off, the last cell of its own conversation, can't nest one.
FRESH_HISTORIAN = {
    "tools": [HISTORY_TOOL, "page"],
    "spawn": [],
    "system_suffix": HISTORIAN_SUFFIX,
}
FRESH_NAMESPACES = {
    ROOT: FRESH_ROOT | {"system_suffix": ROOT_SUFFIX},
    HISTORIAN: FRESH_HISTORIAN,
    LIBRARIAN: FRESH_LIBRARIAN,
}


def read_examples(*files):
    return [
        Path(__file__).with_name(file).read_text(encoding="utf-8") for file in files
    ]


# The historian's worked examples: without them, models answer in the cell that prints
# history. Each opens with the question and root's hand-off.
HISTORIANS_EXAMPLES = read_examples(
    "history_example.yaml",
    "typed_example.yaml",
    "sub_agents_example.yaml",
    "page_example.yaml",
    "table_example.yaml",
    "wrong_place_example.yaml",
    "nothing_example.yaml",
)
# The librarian's; without the worked-example one, models type real <repl> tags in a cell.
LIBRARIANS_EXAMPLES = read_examples(
    "read_library_example.yaml", "worked_example_example.yaml", "undo_example.yaml"
)
# Root's are those two turns of each, under the same name: every namespace inherits them,
# but in historian its own take their place.
HAND_OFF_TURNS = 2
# Root's hand-offs to the librarian, each named as one of its own, which replaces it there.
LIBRARIAN_HAND_OFFS = read_examples(
    "read_library_hand_off.yaml", "change_library_hand_off.yaml"
)
# Why a namespace's agents can't hand a question to *target*, as the panel and the worked
# example shown in place of the hand-off say.
CANT_REACH = jinja2.Template(
    "{% if gone %}there is no {{ target }} namespace now"
    "{% else %}this namespace's spawn list leaves it out{% endif %}",
    undefined=jinja2.StrictUndefined,
)
# The system prompt's fan-out names no namespace, and runs where a spawn list may refuse it.
SPAWN_LINE = jinja2.Template(
    "Sub-agents run only in {{ allowed | join(', ') }} or namespaces under them;"
    " subagent() without namespace= runs in your own namespace, which these may not"
    " include, so name one on every call, fan-outs too: run_all({k:"
    ' anext(subagent(namespace="{{ allowed[0] }}").send(t)) for k, t in {"a": "first'
    ' task", "b": "second task"}.items()}){% if above %}. This replaces any line above'
    " on sub-agents{% endif %}.",
    undefined=jinja2.StrictUndefined,
)
# A task asking for a namespace by name reaches it still asking, and agents that may spawn
# into their own namespace handed it on again; one under another reads that one's line too.
HERE_LINE = jinja2.Template(
    "You are in the {{ name }} namespace{% if above %}, not the one named above{% endif %}"
    '{% if loops %}: a task asking for {{ name }} ("hand this to {{ name }}") is already'
    " where it was sent, so do it here rather than hand it on{% endif %}.",
    undefined=jinja2.StrictUndefined,
)
# Root's suffix says Python runs unsandboxed, as the local REPL does; any other is a sandbox.
LOCAL_REPL = "local"
SANDBOX_LINE = jinja2.Template(
    "Here your Python runs {% if type == local %}unsandboxed, in the local REPL{% else %}"
    "in the {{ type }} REPL, a sandbox{% endif %}, whatever is said above.",
    undefined=jinja2.StrictUndefined,
)
NO_SUB_AGENTS = (
    "Whatever is said above of sub-agents, agents here can't hand work to them: subagent"
    " is refused, so answer without them."
)
# Where the panel and show(name) say the lines above come from: the app, never stored.
ADDED_AT_RUN_TIME = "added by the app at run time"
# The lines above, as their shapes, in text as plain_words() writes it: a stored text
# holding one would pass for the app's own line, as which namespace its agents are in,
# where their Python runs (a sandbox, say) or where they may hand work.
RUN_LINE_SHAPES = re.compile(
    r"\byou(?: are|'re) in the \S+ namespace\b"
    r"|\byour python (?:code )?runs\b"
    r"|\bsub-?agents (?:may )?run only in\b"
    r"|\breplaces any line above\b"
    r"|\bwhatever is said above of sub-?agents\b"
    r"|\bagents here (?:can't|cannot|can not) hand work\b"
    r"|\bsubagent is refused\b"
)
CURLY_APOSTROPHES = "\u2018\u2019\u02bc"
# What a namespace that can't make a worked example's hand-off is shown in its place, by
# kind: "here" when it is the target or under it, so it answers itself; "record" when only
# the target reads the chat's record, so it can't; else it answers itself if it can.
CANT_HAND_OFF = jinja2.Template(
    """\
{% if kind == "here" %}<think>Questions like this are answered in the {{ target }} \
namespace, where I am: I answer it here and never hand it on\
{% if record %}, reading the record with history first{% endif %}.</think>
<repl>
{% if record %}print(history.turns()){% else %}print(ctx.task){% endif %}
</repl>
{% elif kind == "record" %}<think>Questions like this are answered from this chat's \
record, which the {{ target }} namespace reads, but {{ why }}. Redoing the work, guessing, \
or handing it to another namespace would answer something else: say I can't.</think>
<repl>
FinalAnswer({{ answer }})
</repl>
{% else %}<think>Questions like this go to the {{ target }} namespace, but {{ why }}, so I \
can't hand it off: I answer it myself if I can, and say so if I can't.</think>
<repl>
print(ctx.task)
</repl>
{% endif %}""",
    undefined=jinja2.StrictUndefined,
)
CANT_ANSWER = jinja2.Template(
    "I can't answer that here: questions like it go to the {{ target }} namespace, and "
    "{{ why }}.",
    undefined=jinja2.StrictUndefined,
)
# What the panel says such a namespace's agents do with those questions, by kind.
INSTEAD = {
    "here": "answer themselves",
    "record": "say they can't answer",
    "other": "try themselves",
}
UNTOUCHED_ROOT = (
    "name = :root AND yaml = :empty"
    " AND NOT EXISTS (SELECT 1 FROM decompositions WHERE namespace = :root)"
)
SYSTEM_PROMPT_FILE = Path(__file__).with_name("system_prompt.yaml")
DEFAULT_SETTINGS = yaml.safe_load(SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"))
# Each chat picks its model, and the provider that serves it; a Library made before that
# may still hold these, which are left unread.
CHATS_DECIDE = {"model", "client"}
# What an import reads but doesn't store as a run setting: the content, and its own path.
CONTENT = {"namespaces", "namespaces_dir", "decompositions", "tools", "config_path"}
# Left to the Library, not a config: content (own tables), per-run paths, and the model and
# its key. A shared file's run setting of these is left out.
NOT_RUN_SETTINGS = CONTENT | {
    "task",
    "task_id",
    "run_prefix",
    "log_dir",
    "load_dotenv",
    "model",
    "planner_model",
    "models",
    "llm_kwargs",
    "client",
    "reasoner",
}
# Every run has one, so the Library is checked and shown as a run sees it: with the llm
# tool a model brings.
A_MODEL = {"model": "the chat's model"}
# The tools a run declares as it has a model, a Library row of theirs or not: the llm.
BROUGHT_BY_A_MODEL = frozenset(declared_tool_names(V2Config.model_validate(A_MODEL)))
# A tool of this factory, or a reasoner of this type, runs on the user's subscription.
CLAUDE_CODE = "claude_code"
ENTRY_NAMESPACE = "entry_namespace"
# The run settings the panel always lists, with the engine's default when unset.
SHOWN_SETTINGS = ("max_iter", "max_depth", ENTRY_NAMESPACE, "repl")
LINE_WIDTH = 88
NEL = "\x85"
ROLES = "user, assistant or system"
# A worked example's turns, as deep_reasoner's own namespace examples write them (its docs/configs).
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
REPL = "<repl>"
OBSERVATION = re.compile(r"<observation>.*?</observation>", re.DOTALL)
# ASCII, as web/turns.js reads it; a whole-line comment calls nothing.
FINAL_ANSWER = re.compile(r"\bFinalAnswer[ \t]*\(", re.ASCII)
COMMENT_LINE = re.compile(r"[ \t]*#")
BLANK_NAME = 'The name should be text, like "name: count the open tickets".'
BETWEEN = {
    "user": "the assistant's turn goes between them",
    "assistant": "the user turn with the <observation> goes between them",
}
NO_TASK = "There is no task: add a user turn holding it."
# make_tool builds `factory: llm` with v2's own builder, the rest from TOOL_BUILDERS.
BUILDERS = {**TOOL_BUILDERS, "llm": build_llm_tool}
# What a custom tool's source is read for: what it defines, and the Func(...) it returns.
DOCUMENTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNC = "Func"
NO_DESCRIPTION = "no description found in its source"
BUILT_DESCRIPTION = "set when it's built: {}"
# The setting a tool's factory may describe it by, as the engine's own do.
DESCRIBED_BY = "description"
# The engine's factories that take their own description for an empty one too; the
# others keep it empty.
EMPTY_TAKES_DEFAULT = {"kg", "claude_code"}
BUILTIN_DESCRIPTIONS = {
    "llm": _DEFAULT_LLM_DESCRIPTION,
    "rag": DEFAULT_RAG_DESCRIPTION,
    "kg": inspect.getdoc(KnowledgeGraph),
    "safe_url": DEFAULT_SAFE_URL_DESCRIPTION,
    "claude_code": DEFAULT_CLAUDE_DESCRIPTION,
}
HISTORY_DESCRIPTION = """\
When the user asks about an earlier message (how you got an answer, why you did \
something, what an agent did or found, what went wrong), read what was recorded with \
`history`, then answer in a later cell from what it printed, not by redoing the work or \
guessing. The earlier turns, in ctx.history too, hold only each message and its answer; \
history holds every step of the earlier runs. Messages are numbered from 1, the chat's \
first. In each, the top agent (node None) answered, and its cells may have started \
sub-agents, each with a node of its own. The methods return plain data (lists of dicts), \
and the REPL shows only what you print: print what you need.
  history.turns(): each earlier message: turn, message, namespace, model, outcome, \
answer, detail, cost, seconds, agents (how many sub-agents it ran) and failed (how many \
of its cells failed, its sub-agents' included).
  history.agents(turn): each sub-agent drive of message `turn`: node, parent, drive, \
task, namespace, status, result, seconds (from its start to its end, to the tenth), cost \
(its own sub-agents' included), steps (how many) and failed (how many of its cells \
failed).
  history.steps(turn, node=None): the thoughts, cells, Library changes and pages, in \
order, of message `turn`'s top agent, or of the sub-agent whose node is `node`: {kind: \
"think", text}, {kind: "cell", code, output, status ("done", "failed", "stopped", or "cut \
off" by a run that ended otherwise), agents: the nodes of the sub-agents it started}, \
{kind: "change", what, namespace, name, before (None when added), after} or {kind: \
"page", page, title} (a page opened for the user), each with its drive.
  history.errors(turn=None): what went wrong anywhere in message `turn`, or in every \
earlier message: each failed cell of any agent, the top one included, as turn, node, \
namespace (the agent's), task, index (its place in history.steps(turn, node)), error \
(what it raised) and code; then each agent that ended failed, stopped, cut off or \
exhausted, with index None. Start here for any question about an error.
  history.find(text, turn=None): where `text` is, ignoring case, in any agent's task, \
thoughts, code, outputs and result: turn, node, index (None for a task or result), kind \
and a snippet. Use it to find anything the user names.
  history.full(turn, node, index): history.steps(turn, node)[index] whole.
Each text these return is cut after 2,000 characters, ending "…(+N chars)" with how many \
were cut; history.full gives a step uncut. errors and find list at most 50 rows, then \
{more, note}: how many more, and to pass a turn to list fewer."""
LIBRARY_DESCRIPTION = """\
The `library` tool reads and writes the app's Library: its namespaces, each with its own \
fields, the decompositions (worked examples) each one's agents are shown, and the tools \
namespaces may grant. Its methods return plain data, and the REPL shows only what you \
print: print what you inspect. A change applies from the user's next message, not in \
this run, and only a chat whose namespace grants this tool may make one. A namespace's \
agents get what it sets and what the namespaces above it set (root, then research, then \
research.papers, say): their tools, vars and system_suffix texts add up, so a \
namespace's own system_suffix holds only what is new, and a decomposition of the same \
name lower down takes the place of its parent's there. Where a method takes \
yaml, give YAML text or a dict of plain values.
  library.show(name=None): every namespace, in brief: name, depth, sets (the fields it \
sets itself), tools (all its agents get, inherited ones included) and decompositions \
(only those it holds itself: each name, with how many problems it has as a worked \
example: 0 is ✓). Each text is cut after 200 characters, ending "…(+N chars)". \
show(name) gives namespace `name` in full, each text cut only after 20,000 characters \
(show(name, whole=True) cuts none): name, depth, yaml (the fields it sets itself, \
as change_namespace takes them), fields (each as the engine resolves it, with where it \
comes from), decompositions (every one its agents are shown, inherited ones included: \
each with its name and from, the namespace that holds it; those `name` holds itself \
with their yaml and problems too, and show(from) gives an inherited one's; rewritten: \
True on one they are shown doing what they can instead of a hand-off they can't make, \
as the Hand-offs field says) and decompositions_left_out (those it holds that no agent \
is shown, as a shared file's the engine can't read: name, yaml and problems).
  library.tools(name=None): every tool, in brief: name, kind ("custom", "built-in" or \
"the app's"), factory, settings, granted_in (the namespaces that grant it themselves), \
description and source (a custom tool's Python, else None), each text cut after 200 \
characters. tools(name) gives tool `name` whole.
  library.changes(turn=None): what this chat changed in the Library, in each of its \
messages (this one's so far included) or in message `turn` (numbered from 1, the chat's \
first); what other chats changed is listed under their answers, not here: turn, \
message, what (namespace, decomposition or tool), namespace, name, before (None when it \
was added) and after, whole; a namespace's with changed (each field it \
changed, before and after), and a tool's before and after as {source, factory, \
settings}. To undo a change, write its before back, from the value changes() returned, \
never retyped: for a namespace, each field it changed, change_namespace(name, {field: \
value["before"] for field, value in change["changed"].items()}), which keeps what later \
changes set in other fields; write_decomposition(namespace, change["before"]); \
write_tool(name, **change["before"]). A change whose before is None added something, \
and undoing it is deleting it, which this tool never does: the user can, in the Library \
panel (a namespace in its Namespaces tab, a decomposition in its Decompositions tab, a \
tool no namespace grants in its Tools tab).
  library.write_decomposition(namespace, yaml): adds the decomposition written as YAML \
to namespace `namespace`, or replaces the one of that name there. Agents copy it, so \
write it as they work, like those show(name) lists: name: <its name>, then messages:, a \
list of turns, each a role and content (content: | for several lines): the user's task; \
then assistant turns, each <think>…</think> and one <repl>…</repl> of Python; after each \
but the last, a user turn with the <observation>…</observation> its code printed; the \
last calls FinalAnswer(...). The last <repl> block in your reply is the one that runs, \
so in your cell write the example's repl tags as [repl] and [/repl], each on a line of \
its own, in a raw string (r\"\"\"…\"\"\") so its code keeps the escapes you type; in each \
assistant turn, the tool writes a line that is [repl] or [/repl] alone as the real tag, \
and leaves any other [repl], in a user's turn or in code, as you typed it. A yaml that \
already holds a real <repl> tag, as one from show(name) or changes() does, is written \
as it is, [repl] and all: to edit a stored decomposition, change its yaml in your cell \
(with .replace(...)) rather than typing it again. It is stored even when it isn't yet \
a worked example like that: the result's problems lists each way it isn't (empty when \
none), and until you write it again without them the Library panel marks it ✗. The \
result's code_problems, when there are any, lists each turn whose code isn't valid \
Python, which agents copying it would copy too.
  library.add_namespace(name, yaml): adds namespace `name` ("research", or \
"research.papers" under an existing research) with the fields yaml sets: system_suffix \
(text added to its agents' system prompt), vars (names and values its agents' Python \
has), tools (tool names it grants) and spawn (the namespaces its agents may hand work \
to; [] for none).
  library.change_namespace(name, yaml, replace=False): changes only the fields yaml \
sets, keeping the others; a field set to null (system_suffix: null) is taken out. A \
field yaml sets takes the place of the namespace's own whole: tools, spawn and vars \
too, so to grant one more tool give the whole list (the namespace's own tools, as \
show(name)'s yaml gives them, and the new one), and to add a var give all its vars. \
With replace=True, yaml is the namespace's whole own YAML, and each field it doesn't set \
is taken out. It keeps the namespace's decompositions, and returns changed (each field \
it changed, before and after) and kept (the fields it left as they were).
  library.write_tool(name, source, factory, settings=None): adds the tool `name` (a \
Python name), or replaces the one of that name, from `source`, the text of a Python \
module that defines the function `factory`. A run calls factory(client, params), with \
params = `settings` (a dict of plain values), and it returns \
deep_reasoner.primitives.Func(value, description=...): value is what agents call by the \
tool's name (a function, or an object whose methods they call), and description is what \
they are told it does (value's docstring when not given; name the methods). client is \
the run's async OpenAI client: client.chat.completions.create(...) returns a coroutine, \
not a reply. A tool that needs the model takes the agent's llm as an argument and calls \
llm(prompt), which returns text from the chat's model; never name a model in a tool. It \
is built once here to check it, as another chat would build it: in a process of its \
own, from an empty folder, so a tool that needs a file or module only this chat's \
folder has doesn't build. Building it doesn't call it: say it was built, not tried. \
Agents see it only in a namespace that grants it (add its name to a namespace's tools \
with change_namespace), but once any namespace grants it, its module and factory run at \
the start of every run of every chat, so add_namespace and change_namespace first build \
each custom tool they grant that no namespace granted yet, and refuse one that doesn't \
build, as it would stop every chat. Writing or granting a custom tool works only in a \
chat whose namespace runs the local REPL.
Locked, so you may read them but not write them: root; a namespace's repl and reasoner; \
which namespaces grant the app's own tools (library, history, page); granting a tool \
that runs Claude Code (its factory is claude_code), which bills the user's subscription; \
a new namespace under one that grants library or runs Claude Code (as its reasoner or a \
tool); a tool root grants, or one the engine builds from its block alone (no Python of \
its own); a tool named as a built-in (llm, rag, kg, safe_url, claude_code), one of the \
app's, a Python keyword or a name the REPL binds itself. The user changes any \
namespace, root included, in the Library panel's Namespaces tab (Edit YAML): what every \
chat should do goes in root's system_suffix. There is no method for the system prompt \
or the run settings (max_iter, max_depth, ...): only the user changes them, by importing \
a config (deep-reasoning import).
Each write returns what it did, as {done, what, name} (and namespace, but for a tool); \
a refusal raises ValueError saying why. One that holds "…(+N chars)", where a text \
was cut short, or "…(+N more items)", where a long list was, is refused: write the \
whole text."""
PAGE_DESCRIPTION = """\
When the user asks to see something (how an earlier answer was worked out, what an \
agent did, an error) and you found it, show it on a page with `page`: the app opens it \
in a tab of its own, linked under your answer. When you found nothing to show, say so in \
words and make no page. You pick what goes on it and the app draws it, copying each \
step and sub-agent from the record, so write no HTML. Its methods take plain data (text, \
numbers, True, False, None, and lists and dicts of them), and the REPL shows only what \
you print.
  p = page.new(title): a page in progress, titled `title` (up to 200 characters); \
nothing is saved until p.open().
  p.text(markdown): a paragraph or more of Markdown.
  p.step(turn, node=None, index=i, note=None): history.steps(turn, node)[i], drawn as \
the chat draws it, with an optional Markdown note.
  p.agent(turn, node, note=None): the sub-agent whose node history.agents(turn) lists, \
drawn as a tree with its steps and its own sub-agents', with an optional note.
  p.table(rows, columns=None): rows as a table: a list of dicts (the columns are their \
keys, unless given) or of lists, each value text, a number, True, False or None.
  print(p.open()): saves the page and says so, and the chat links it under your answer; \
opened again, it is saved as it stands then, on the same page.
Never write the page's link yourself, and still say in your answer what you found. step \
and agent take a turn, node and index as history shows them, so they reach this chat's \
earlier messages only. A page holds up to 200 blocks, each text or note up to 100,000 \
characters and each table up to 1,000 rows of 50 columns, and up to 2,000,000 characters \
in all. A refusal raises ValueError saying why."""
# Added in code, never stored: a shared file mustn't choose the code they run. Each one's
# factory in app_tools.py, and what the agent is told it does.
APP_TOOLS_FILE = Path(__file__).with_name("app_tools.py")
APP_TOOLS = {
    "history": ("build_history", HISTORY_DESCRIPTION),
    LIBRARY_TOOL: ("build_library", LIBRARY_DESCRIPTION),
    "page": ("build_page", PAGE_DESCRIPTION),
}
APP_TOOLS_MODULE = f"{__package__}.{APP_TOOLS_FILE.stem}"
APP = "Deep Reasoning"
# No tool of your own may have one of these names, nor a Python keyword's; a config may
# give a built-in its settings under the built-in's name.
TAKEN_NAMES = (*BUILDERS, *APP_TOOLS, *sorted(RESERVED_NAMES))
# A new namespace's name, from the page: plain words, joined by dots for a child.
NAMESPACE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*)*")
NAME_IT = 'Name the namespace, like "research"; "research.papers" goes under research.'
NAME_SHAPE = (
    "A namespace name is words of letters, digits, _ and -, each starting with a letter,"
    ' joined by dots, like "research.papers".'
)
# What a namespace's YAML leaves to the panel: its name field, and the Decompositions tab.
NAMED_ELSEWHERE = {
    "name": 'Remove "name": a namespace is named in the name field, not in its YAML.',
    "decompositions": (
        'Remove "decompositions": a namespace\'s decompositions are added'
        " in the Decompositions tab."
    ),
}
NAMESPACE_KEYS = [
    key for key in NamespaceConfig.model_fields if key not in NAMED_ELSEWHERE
]
NAMESPACE_HOLDS = f"{', '.join(NAMESPACE_KEYS[:-1])} and {NAMESPACE_KEYS[-1]}"
SHAPES = {
    "tools": 'tools should be a list of tool names, like "tools: [llm]".',
    "spawn": 'spawn should be a list of namespace names, like "spawn: [research]", or [] for none.',
    "vars": 'vars should be a mapping of names to values, like "vars: {year: 2026}".',
    "system_suffix": 'system_suffix should be text; write "system_suffix: |" with the text indented below.',
    "reasoner": 'reasoner should be a mapping, like "reasoner: {type: claude_code}".',
}
# How a namespace that sets nothing is kept by an import, or a Library made before
# FRESH_ROOT; the panel saves one as blank text, so a root saved blank is never upgraded.
EMPTY_NAMESPACE = "{}\n"
# The settings V2Config reads; the panel may delete any other, as an import may bring junk.
ENGINE_SETTINGS = V2Config.model_fields.keys()
UNREAD_SETTING = "set in the Library; the engine doesn't read it"
# The backbones the engine builds an agent on; a namespace naming another fails every run there.
BACKBONES = (DeepReasoner.backbone_type, ClaudeDeepReasoner.backbone_type)
# As the library tool's limit; past 64 MiB, a change event would end the run that wrote it.
MAX_YAML_CHARS = 1_000_000
# Why a shared file's row is left out, when it is none the app could have written. A row
# is read only within MAX_ROW_BYTES, as UTF-8 takes at most 4 bytes a character.
MAX_ROW_BYTES = 4 * MAX_YAML_CHARS
TOO_LONG_ROW = f"It is longer than the {MAX_YAML_CHARS:,} characters the Library holds."
# How much YAML may stand for with its aliases written out, as values plus their text's
# characters (expanded_size()): YAML written out never stands for twice its characters,
# and aliases may stand for this much more.
ALIASES_STAND_FOR = 2
ALIASES_ALLOWANCE = 10_000
YAML_NOT_TEXT = "Its YAML isn't text."
GRANT_IGNORED = (
    "Runs ignore its grants until it is fixed; Delete takes it out of their tools too."
)
SHADOWS_AN_APP_TOOL = (
    "It has the name of the app's own {} tool, which runs build in its place: Delete"
    " removes only this row."
)
NAME_NOT_TEXT = "Its name isn't text."
NOT_A_RUN_SETTING = "The Library never holds it: each chat or run sets it, or the Library's own tables do."
# What heads the problems of a decomposition left out: one the engine can't read, or
# can't render, which would fail every run in its namespace and below.
UNREAD_DECOMPOSITION = "Agents aren't shown it: the engine can't read it."
UNRENDERED_DECOMPOSITION = "Agents aren't shown it: a run can't render it."
UNSAVED_DECOMPOSITION = "Agents aren't shown it: no save would store it so."
# How many stored decompositions' readings are kept, so a read of the Library parses only
# those that changed; a Library holds a few dozen.
DECOMPOSITIONS_KEPT = 256
FIELDS_MAPPING = 'The YAML should be a mapping of fields, like "tools: [llm]".'
# A var named as one of these hides it from the agent's code, as max: 10 breaks max(numbers).
BUILTIN_NAMES = frozenset(dir(builtins))
# Modules the agent's code commonly uses, and the names it imports some as: a tool named
# as one hides it, until the code imports it, which hides the tool.
COMMON_MODULES = frozenset(
    {
        *("asyncio", "base64", "collections", "copy", "csv", "datetime", "decimal"),
        *("functools", "glob", "hashlib", "heapq", "io", "itertools", "json"),
        *("logging", "math", "operator", "os", "pathlib", "pickle", "random", "re"),
        *("shutil", "sqlite3", "statistics", "string", "subprocess", "sys"),
        *("tempfile", "textwrap", "time", "typing", "urllib", "uuid", "zipfile"),
        *("numpy", "np", "pandas", "pd", "scipy", "matplotlib", "plt", "seaborn"),
        *("sns", "sklearn", "sympy", "networkx", "nx", "polars", "requests", "httpx"),
        *("yaml", "bs4", "PIL"),
    }
)
# Alphabets with letters that pass for one another's, as Cyrillic а for Latin a.
LOOK_ALIKE_SCRIPTS = {"LATIN", "CYRILLIC", "GREEK", "ARMENIAN"}
# Characters that don't show: controls and formats, such as a zero-width space.
INVISIBLE = ("Cc", "Cf")
# What the engine renders a turn's template against, beside prompt_template_variables.
FILLED_IN = ("task", "vars", "funcs", "coroutines")
# Never rendered, only parsed and compiled; with no globals, range() is an unknown name.
JINJA = Environment()
JINJA.globals.clear()
TEMPLATE = "the engine reads text holding {{ or {% as a Jinja template"
ESCAPE = (
    "Write {{ '{{' }} for a literal {{ and {{ '{%' }} for {%, or put the text between"
    " {% raw %} and {% endraw %}."
)
NAMES_ONLY = (
    "a template may name only task, vars, funcs, coroutines and the run settings'"
    " prompt_template_variables, which the engine fills in"
)
# An empty loader, so an include fails as a render error the engine words.
SANDBOXED = partial(SandboxedEnvironment, loader=DictLoader({}))
INCLUDES = (nodes.Include, nodes.Extends, nodes.Import, nodes.FromImport)
SCREENED = (*INCLUDES, nodes.Getattr, nodes.Getitem)
# The names the sandbox refuses that don't start with _.
UNSAFE_NAMES = frozenset(
    {"mro"}.union(
        UNSAFE_FUNCTION_ATTRIBUTES,
        UNSAFE_METHOD_ATTRIBUTES,
        UNSAFE_GENERATOR_ATTRIBUTES,
        UNSAFE_COROUTINE_ATTRIBUTES,
        UNSAFE_ASYNC_GENERATOR_ATTRIBUTES,
    )
)
# Compiling a template nested too deep raises Python's own errors, not Jinja's.
TEMPLATE_FAILURES = (TemplateError, SyntaxError, RecursionError, MemoryError)


class LibraryError(ValueError):
    """Something the Library refused, in words for the person who asked."""


class ChangedMeanwhile(LibraryError):
    """A save or delete refused because what it was opened on changed since."""


# What a change nobody opened in the panel, as the librarian's, is checked against: nothing.
NOT_OPENED = object()
CHANGED_MEANWHILE = (
    "The tools this grants changed while they were being built; try again."
)
# SQLite's names for a file it can't write: read only, full, failing, or gone.
UNWRITABLE = ("SQLITE_READONLY", "SQLITE_FULL", "SQLITE_IOERR", "SQLITE_CANTOPEN")


BY_WHOM = "(by the librarian, another tab or an import)"
# Half of a character, which a YAML escape can make (PyYAML reads an emoji's escaped pair
# as two halves) but no text written as UTF-8, so no response or run's spec, can hold.
SURROGATE = re.compile("[\ud800-\udfff]")


def unwritable(error):
    """What the SQLite *error* says, in words, when it is that the Library file can't be
    written; None when it is anything else."""
    if getattr(error, "sqlite_errorname", "").startswith(UNWRITABLE):
        return f"The Library file can't be written: {error}."
    return None


def changed_meanwhile(what):
    return ChangedMeanwhile(
        f"{what} was changed after you opened it {BY_WHOM}, so nothing was saved:"
        " saving would undo that change. Look at it as it is now, then save again."
    )


class BlockDumper(yaml.SafeDumper):
    """Writes multi-line text as | blocks (text holding a NEL double-quoted), so stored YAML reads like the engine's configs."""


def represent_text(dumper, text):
    # PyYAML escapes a NEL only in double quotes, and reads a raw one as a line break.
    style = '"' if NEL in text else "|" if "\n" in text else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", text, style=style)


BlockDumper.add_representer(str, represent_text)


def dump(value):
    return yaml.dump(
        value, Dumper=BlockDumper, sort_keys=False, allow_unicode=True, width=LINE_WIDTH
    )


def flow(value):
    one_line = yaml.safe_dump(
        value,
        default_flow_style=True,
        sort_keys=False,
        allow_unicode=True,
        width=sys.maxsize,
    )
    return one_line.strip()


@contextmanager
def writing(connection):
    """One transaction that takes the write lock up front, so concurrent writers wait their turn instead of failing."""
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        yield


@contextmanager
def reading(connection):
    """One read transaction, so what is read in it is the Library as it stood at one moment."""
    with connection:
        connection.execute("BEGIN")
        yield


def schema_version(connection):
    return connection.execute("PRAGMA user_version").fetchone()[0]


def open_library(path):
    """The Library at *path*; a new file starts with a root namespace granting llm and the generic REPL prompt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Autocommit: a read holds no lock once it has run; writes go through writing().
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        # Deleted rows are wiped, not left in free pages that a copy would carry.
        connection.execute("PRAGMA secure_delete = ON")
        distrust(connection)
        if schema_version(connection) != SCHEMA_VERSION:
            lay_out(connection, path)
        freshen_an_untouched_root(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def distrust(connection):
    """Never run the file's own triggers and views: a shared file's could keep what is
    deleted. Read text that isn't UTF-8, as a shared file's may be, with U+FFFD for each
    byte that can't be read, never as an error."""
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, False)
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_VIEW, False)
    connection.text_factory = readable


def readable(data):
    return data.decode(errors="replace")


def lay_out(connection, path):
    """Write the whole app file into an empty file, or add what a file of an earlier version
    lacks, keeping what it holds; refuse, untouched, a file that holds anything else."""
    with writing(connection):
        # Read again under the lock: another process may have just laid it out.
        found = schema_version(connection)
        if found == SCHEMA_VERSION:
            return
        names = {
            name for (name,) in connection.execute("SELECT name FROM sqlite_master")
        }
        if not written_by_an_earlier_version(found, names):
            raise LibraryError(
                f"{path} is not a Library this version of Deep Reasoning can open."
            )
        versions = range(found + 1, SCHEMA_VERSION + 1)
        added = [each for version in versions for each in ADDED_IN[version][1]]
        for statement in added:
            connection.execute(statement)
        if found == 0:
            seed_new_library(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def written_by_an_earlier_version(found, names):
    """Whether a file at version *found*, holding *names*, is empty or one an earlier
    version wrote: every table of its version, and none of a later one's."""
    if found == 0:
        return not names
    return 0 < found < SCHEMA_VERSION and all(
        tables <= names if version <= found else not tables & names
        for version, (tables, _) in ADDED_IN.items()
    )


def seed_new_library(connection):
    for name, fields in FRESH_NAMESPACES.items():
        connection.execute(UPSERT_NAMESPACE, (name, dump(fields)))
    examples = [(ROOT, text) for text in ROOTS_EXAMPLES]
    examples += [(HISTORIAN, text) for text in HISTORIANS_EXAMPLES]
    examples += [(LIBRARIAN, text) for text in LIBRARIANS_EXAMPLES]
    rows = [(namespace, parsed(text).name, text) for namespace, text in examples]
    connection.executemany(UPSERT_DECOMPOSITION, rows)
    settings = [(key, dump(value)) for key, value in DEFAULT_SETTINGS.items()]
    connection.executemany(UPSERT_SETTING, settings)


def hand_off(text):
    """Root's worked example from the historian's written as YAML *text*: its question and
    root's hand-off, under its name."""
    example = parsed(text).model_dump()
    return dump(example | {"messages": example["messages"][:HAND_OFF_TURNS]})


def freshen_an_untouched_root(connection):
    """Give a Library made with an empty root the root a new one starts with."""
    parameters = {"root": ROOT, "empty": dump({}), "fresh": dump(FRESH_ROOT)}
    found = connection.execute(
        f"SELECT 1 FROM namespaces WHERE {UNTOUCHED_ROOT}", parameters
    )
    if not found.fetchone():
        return
    try:
        with writing(connection):
            connection.execute(
                f"UPDATE namespaces SET yaml = :fresh WHERE {UNTOUCHED_ROOT}",
                parameters,
            )
    except sqlite3.OperationalError as error:
        # A file it can't write, such as a read-only copy, still opens as it is.
        log.warning("library.root_not_upgraded", error=str(error))


def tool_file(name):
    """A custom tool's file name; a Python name keeps it inside the folder it is written to."""
    if not name.isidentifier():
        raise LibraryError(f'Tool "{name}" needs a Python name, like "waitlist".')
    return f"{name}.py"


def parsed_python(source, what="The source"):
    """The Python *source* parsed, or a LibraryError saying why it doesn't parse."""
    try:
        return ast.parse(source)
    except SyntaxError as error:
        raise LibraryError(
            f"{what} is not valid Python (line {error.lineno}: {error.msg})."
        ) from error
    except (RecursionError, MemoryError) as error:
        raise LibraryError(f"{what} is nested too deeply to parse.") from error


def tool_text(name, text, source):
    """Tool *name*'s row as a change shows it: its block's YAML *text*, then the Python
    of the file it names."""
    return f"{text}# {tool_file(name)}\n{source}"


def fitting(column):
    """SQL reading *column* only when its bytes could hold MAX_YAML_CHARS characters, then
    its size in bytes, NULL for NULL: length() of text would stop at a NUL."""
    size = f"length(CAST({column} AS BLOB))"
    return f"CASE WHEN {size} <= {MAX_ROW_BYTES} THEN {column} END, {size}"


def row_text(text, size):
    """A row's *text*, *size* bytes, as fitting() read it; a LibraryError saying why it
    can't be read."""
    if type(text) is str and len(text) <= MAX_YAML_CHARS:
        return text
    if type(text) is str or text is None and size is not None:
        raise LibraryError(TOO_LONG_ROW)
    raise LibraryError(YAML_NOT_TEXT)


def shown_text(text, size):
    """A row's text as the panel shows it, from what fitting() read: None for one too long
    or not text."""
    try:
        return row_text(text, size)
    except LibraryError:
        return None


def warn_left_out(table, key, why):
    """*why* the row *key* of *table* is left out, logged."""
    log.warning("library.row_left_out", table=table, row=repr(key), why=why)
    return why


def left_out_note(why):
    """What a change logs as the before of a row the engine can't read, in place of its
    text, which may be too long or not text: why, as YAML comments."""
    return "".join(f"# {line}\n" for line in f"Left out: {why}".splitlines())


def unreadable(error):
    """A ValidationError of the engine's config model as a LibraryError, in its words."""
    first = error.errors()[0]
    where = ".".join(map(str, first["loc"]))
    return LibraryError(f"The engine can't read it ({where}: {first['msg']}).")


def setting_value(key, text):
    """Run setting *key*'s value, from its YAML *text*, if the engine reads it so; else a
    LibraryError saying why not."""
    if key in NOT_RUN_SETTINGS:
        raise LibraryError(NOT_A_RUN_SETTING)
    value = loaded(text)
    try:
        V2Config.model_validate({key: value})
    except ValidationError as error:
        raise unreadable(error) from error
    return value


def read_run_settings(connection, namespaces):
    """The run settings that load, as {key: value}, and those left out, as {key: why}.
    Each chat picks its model and client, so an older file's go unread. An
    entry_namespace naming neither root nor one of *namespaces*, those the engine reads,
    is left out too, as the engine would refuse every run and every write."""
    settings, left = {}, {}
    query = f"SELECT key, {fitting('yaml')} FROM settings ORDER BY rowid"
    for key, text, size in connection.execute(query):
        if type(key) is not str:
            warn_left_out("settings", key, NAME_NOT_TEXT)
            continue
        if key in CHATS_DECIDE:
            continue
        try:
            settings[key] = setting_value(key, row_text(text, size))
        except LibraryError as error:
            left[key] = warn_left_out("settings", key, str(error))
    entry = settings.get(ENTRY_NAMESPACE, ROOT)
    if entry != ROOT and entry not in namespaces:
        del settings[ENTRY_NAMESPACE]
        why = f'It names namespace "{entry}", which the engine doesn\'t read.'
        left[ENTRY_NAMESPACE] = warn_left_out("settings", ENTRY_NAMESPACE, why)
    return settings, left


def stored_settings(connection):
    return read_run_settings(connection, read_namespaces(connection)[0])[0]


def check_tool_block(name, block):
    """Raise unless tool *name*'s config *block* is one the Library may hold: factory and
    factory_from text; without factory_from, one of the engine's own factories; with it,
    a function of the tool's own Python, never named as one of the engine's, so no block
    reads as one of them (claude_code runs on the user's subscription)."""
    factory, factory_from = block.get("factory", name), block.get("factory_from")
    if type(factory) is not str or not isinstance(factory_from, str | None):
        raise LibraryError(f'Tool "{name}": factory and factory_from should be text.')
    if factory_from is None and factory not in BUILDERS:
        raise LibraryError(
            f'Tool "{name}" names factory "{factory}", which is not built in; give its'
            " block a factory_from: file."
        )
    if factory_from is not None and factory in BUILDERS:
        raise LibraryError(
            f'Tool "{name}" names factory "{factory}", one of the engine\'s own'
            f" ({', '.join(BUILDERS)}), from Python of its own: name that function"
            " otherwise, so the tool never reads as the engine's."
        )


def check_tool_name(name, block):
    """Raise unless a tool of the Library may be named *name*, as its config *block*
    makes it: a Python name (tool_file()), as agents call it by its name; none taken
    (taken()), but by a built-in's own block, which gives that built-in its settings; and
    none hiding one of Python's (tool_name_problem())."""
    tool_file(name)
    the_builtin = (
        block.get("factory_from") is None
        and block.get("factory", name) == name
        and name in BUILDERS
    )
    if taken(name) and not the_builtin:
        raise LibraryError(
            f'Tool "{name}" has a name a built-in, the app or the REPL uses (such as llm,'
            " history or subagent), or a Python keyword, so it would take that one's"
            " place in every namespace. Rename it; a block may use a built-in's name"
            " only to give that built-in its settings."
        )
    if problem := tool_name_problem(name):
        raise LibraryError(problem)


def tool_block(name, text, source):
    """Tool *name*'s block, from its YAML *text*, if it is one the app stores: with its
    Python *source*, or without for one the engine builds; else a LibraryError saying
    why not."""
    block = loaded(text)
    try:
        V2Config.model_validate({"tools": {name: block}})
    except ValidationError as error:
        raise unreadable(error) from error
    check_tool_block(name, block)
    check_tool_name(name, block)
    if (source is None) != (block.get("factory_from") is None):
        raise LibraryError(
            "A tool holds Python exactly when its block names a file for it"
            " (factory_from); this one has one without the other."
        )
    if not isinstance(block.get("description", ""), str):
        raise LibraryError("Its description should be text.")
    return block


def read_tools(connection):
    """The Library's own tools that load, as {name: (block, source)}, and those left out,
    as {name: why}. A row under an app tool's name, as a shared file may hold, is left
    out too: the app's own takes its place, and keeps its grants."""
    query = (
        f"SELECT name, {fitting('yaml')}, {fitting('source')} FROM tools ORDER BY rowid"
    )
    tools, left = {}, {}
    for name, text, size, source, source_size in connection.execute(query):
        if type(name) is not str:
            warn_left_out("tools", name, NAME_NOT_TEXT)
            continue
        if name in APP_TOOLS:
            left[name] = warn_left_out("tools", name, SHADOWS_AN_APP_TOOL.format(name))
            continue
        try:
            source = None if source_size is None else row_text(source, source_size)
            tools[name] = tool_block(name, row_text(text, size), source), source
        except LibraryError as error:
            left[name] = warn_left_out("tools", name, str(error))
    return tools, left


def library_tools(connection):
    return read_tools(connection)[0]


def own_fields(name, text):
    """Namespace *name*'s own fields, from its YAML *text*, if the engine reads them so;
    else a LibraryError saying why not. Its name and decompositions are the app's to put
    in."""
    data = fields_in(text)
    fields = {key: value for key, value in data.items() if key not in NAMED_ELSEWHERE}
    try:
        NamespaceConfig.model_validate(fields | {"name": name})
    except ValidationError as error:
        raise LibraryError("\n".join(map(namespace_problem, error.errors()))) from error
    check_no_run_lines(fields)
    return fields


def read_namespaces(connection):
    """The namespaces that load, as {name: own fields}; those left out, as {name: why};
    and each one's own YAML as the panel shows it, None for one too long or not text.
    One named so it could pass for another (check_stored_name()) goes too. One under a
    namespace left out or not there goes too, as the engine can't resolve it; root's
    stay, as the engine supplies a bare root for a file without one."""
    query = f"SELECT name, {fitting('yaml')} FROM namespaces ORDER BY rowid"
    namespaces, left, shown = {}, {}, {}
    for name, text, size in connection.execute(query):
        if type(name) is not str:
            warn_left_out("namespaces", name, NAME_NOT_TEXT)
            continue
        shown[name] = None
        try:
            shown[name] = row_text(text, size)
            check_stored_name(name)
            namespaces[name] = own_fields(name, shown[name])
        except LibraryError as error:
            left[name] = warn_left_out("namespaces", name, str(error))
    # Parents first, as each has a dot fewer.
    for name in sorted(namespaces, key=lambda each: each.count(".")):
        parent = parent_of(name)
        if parent in (None, ROOT) or parent in namespaces:
            continue
        del namespaces[name]
        gone = "is left out" if parent in left else "is not in the Library"
        why = f'The namespace it goes under, "{parent}", {gone}.'
        left[name] = warn_left_out("namespaces", name, why)
    return namespaces, left, shown


def read_decompositions(connection, names):
    """Each decomposition row, in the order they were added, as {namespace, name, yaml,
    decomposition, why}: yaml None for one the panel can't show, and decomposition, as
    the engine's model reads it, None for one left out, with why (decomposition_read())."""
    query = (
        f"SELECT namespace, name, {fitting('yaml')} FROM decompositions ORDER BY rowid"
    )
    rows = []
    for namespace, name, text, size in connection.execute(query):
        if type(namespace) is not str or type(name) is not str:
            warn_left_out("decompositions", (namespace, name), NAME_NOT_TEXT)
            continue
        decomposition, why = decomposition_read(text, size, name, names)
        if why:
            warn_left_out("decompositions", (namespace, name), why)
        row = {"namespace": namespace, "name": name, "yaml": shown_text(text, size)}
        rows.append(row | {"decomposition": decomposition, "why": why})
    return rows


@lru_cache(maxsize=DECOMPOSITIONS_KEPT)
def decomposition_read(text, size, name, names):
    """A stored decomposition, read from its row's *text* and *size* (as fitting() reads
    them) and the *name* the row is stored under, and why it is left out: the
    decomposition and None when agents can be shown it; None and why, as the panel lists
    it, when the engine can't read it, a save wouldn't store it so, or a run can't render
    one of its turns, whose templates may read only *names*. Kept for the next read of
    the Library, which most requests and every run make."""
    try:
        decomposition = decomposition_in(row_text(text, size))
    except LibraryError as error:
        return None, f"{UNREAD_DECOMPOSITION}\n{error}"
    try:
        check_saved(decomposition)
        if decomposition.name != name:
            raise LibraryError(
                f"It is stored as {name!a}, but its YAML names it"
                f" {decomposition.name!a}, so the panel would list one and agents be"
                " shown the other: the app stores each under its own name."
            )
    except LibraryError as error:
        return None, f"{UNSAVED_DECOMPOSITION}\n{error}"
    if problems := template_problems(decomposition, names):
        return None, "\n".join([UNRENDERED_DECOMPOSITION, *problems])
    return decomposition, None


def read_library(connection):
    """Every row of the Library's tables as the readers above check it, read once: what
    loads, and what is left out with why, by table."""
    namespaces, namespaces_left, namespace_yaml = read_namespaces(connection)
    settings, settings_left = read_run_settings(connection, namespaces)
    tools, tools_left = read_tools(connection)
    names = template_names(settings.get("prompt_template_variables", {}))
    return {
        "settings": settings,
        "settings_left": settings_left,
        "tools": tools,
        "tools_left": tools_left,
        "namespaces": namespaces,
        "namespaces_left": namespaces_left,
        "namespace_yaml": namespace_yaml,
        "decompositions": read_decompositions(connection, names),
    }


def config_of(read):
    """The Library *read* (read_library()) as a plain dict in the engine's config shape,
    the app's tools after its own; a custom tool's factory_from names its stored file.
    A grant of a tool left out is left out too, with a warning, as the engine would
    refuse every run and every write; the llm a model brings stays granted."""
    tools = {name: block for name, (block, _) in read["tools"].items()}
    tools |= {
        name: {"factory_from": str(APP_TOOLS_FILE), "factory": factory}
        for name, (factory, _) in APP_TOOLS.items()
    }
    decompositions = defaultdict(list)
    for row in read["decompositions"]:
        if row["decomposition"] is not None:
            decompositions[row["namespace"]].append(row["decomposition"].model_dump())
    ignored = grants_ignored(read["tools_left"])
    namespaces = {
        name: without_grants(name, fields, ignored)
        | {"decompositions": decompositions[name]}
        for name, fields in read["namespaces"].items()
    }
    return {**read["settings"], "tools": tools, "namespaces": namespaces}


def grants_ignored(left):
    """The tools whose grants runs ignore, of those left out (*left*, by name, as
    read_tools() gives them): all but the llm a model brings and an app tool, whose own
    the app builds in its place."""
    return left.keys() - BROUGHT_BY_A_MODEL - APP_TOOLS.keys()


def without_grants(name, fields, ignored):
    """Namespace *name*'s own *fields* but for its grants of the tools *ignored*, each
    logged."""
    tools = fields.get("tools", [])
    for tool in set(tools) & ignored:
        log.warning("library.grant_ignored", namespace=name, tool=tool)
    if not set(tools) & ignored:
        return fields
    return fields | {"tools": [tool for tool in tools if tool not in ignored]}


def stored_config(connection):
    """The Library as a plain dict in the engine's config shape (config_of()), every row
    the engine can't read, as a shared file may hold, left out."""
    return config_of(read_library(connection))


def grants_of(namespaces, tool):
    """Those of *namespaces*, by name their own fields, that grant *tool* themselves, each
    after the ones above it: as stored, a grant config_of() ignores included."""
    granting = [
        name for name, fields in namespaces.items() if tool in fields.get("tools", [])
    ]
    return sorted(granting, key=chain_of)


def granted(namespaces):
    """Every tool name some namespace grants itself; a list, as a shared file's may be anything."""
    lists = [fields.get("tools") for fields in namespaces.values()]
    return [tool for each in lists if isinstance(each, list) for tool in each]


def handed_to(decomposition):
    """The namespace a worked example hands the whole question to: its last turn's cell
    ending FinalAnswer(subagent(..., namespace="…")) or switch_namespace("…", …), as root's
    hand-offs to the historian do. Only that last statement is read: a subagent call
    before it, or one whose answer the cell works on, delegates a part, which is the
    user's to keep. None too for a name no namespace could have."""
    last = decomposition.messages[-1]
    if last.role != "assistant":
        return None
    try:
        body = parsed_python(running_code(last.content)).body
    except LibraryError:
        return None
    target = (
        hand_off_target(body[-1].value)
        if body and isinstance(body[-1], ast.Expr)
        else None
    )
    return (
        target if isinstance(target, str) and NAMESPACE_NAME.fullmatch(target) else None
    )


def calls(node, name):
    return isinstance(node, ast.Call) and getattr(node.func, "id", None) == name


def hand_off_target(call):
    """The literal namespace *call*, a cell's last expression, hands the question to: as
    FinalAnswer(subagent(..., namespace=X)) or switch_namespace(X, ...)."""
    if calls(call, "FinalAnswer") and call.args and calls(call.args[0], "subagent"):
        named = keyword_value(call.args[0], "namespace")
    elif calls(call, "switch_namespace"):
        named = call.args[0] if call.args else keyword_value(call, "namespace")
    else:
        return None
    return getattr(named, "value", None)


def keyword_value(call, name):
    return next((each.value for each in call.keywords if each.arg == name), None)


def hand_off_targets(config):
    """Where each stored worked example hands off, by its namespace and name: parsed
    once, for the namespaces that inherit it to look up."""
    return {
        (name, example.name): handed_to(example)
        for name, namespace in config.namespaces.items()
        for example in namespace.decompositions
    }


def unreachable(registry, namespace, target):
    """Why *namespace*'s agents can't hand a question to namespace *target*; None when
    they can."""
    if target not in registry.names:
        return CANT_REACH.render(gone=True, target=target)
    if not spawnable(registry, namespace, target):
        return CANT_REACH.render(gone=False, target=target)
    return None


def spawnable(registry, namespace, target):
    """Whether *namespace*'s agents may hand work to namespace *target*, as the engine's
    check_spawn decides."""
    try:
        check_spawn(registry, namespace, target)
    except NamespaceSpawnError:
        return False
    return True


def spawn_lines(registry):
    """What the agents of each namespace that sets a spawn list, and of those under it
    that set none, are told of where they can hand work, by that namespace."""
    lines = {}
    for name in registry.names:
        resolved = registry.resolve(name)
        if resolved.spawn is None or resolved.sources.spawn != name:
            continue
        allowed = [each for each in resolved.spawn if each in registry.names]
        chain = registry.chain(name)
        above = len(chain) > 1 and registry.resolve(chain[-2]).spawn is not None
        lines[name] = (
            SPAWN_LINE.render(allowed=allowed, above=above)
            if allowed
            else NO_SUB_AGENTS
        )
    return lines


def here_line(registry, name):
    """What the agents of namespace *name* are told of which namespace they are in; None
    in root."""
    if name == ROOT:
        return None
    return HERE_LINE.render(
        name=name,
        above=len(registry.chain(name)) > 2,
        loops=spawnable(registry, name, name),
    )


def sandbox_line(registry, name):
    """What the agents of namespace *name* are told of where their Python runs, when its
    REPL is another than the one above it, or than the local REPL root's suffix speaks of;
    None otherwise."""
    chain = registry.chain(name)
    above = registry.resolve(chain[-2]).repl.type if len(chain) > 1 else LOCAL_REPL
    repl = registry.resolve(name).repl.type
    if repl == above:
        return None
    return SANDBOX_LINE.render(type=repl, local=LOCAL_REPL)


def run_lines(registry):
    """What the agents of each namespace are told at run time, after its own
    system_suffix, by namespace, for those told any: which namespace they are in, where
    their Python runs, and where they can hand work."""
    spawns = spawn_lines(registry)
    parts = {
        name: [
            here_line(registry, name),
            sandbox_line(registry, name),
            spawns.get(name),
        ]
        for name in registry.names
    }
    return {
        name: " ".join(filter(None, each)) for name, each in parts.items() if any(each)
    }


def hand_off_kind(registry, namespace, target):
    """What *namespace*'s agents do with a question they can't hand to *target*:
    "here" when it is *target* or under it, "record" when *target* reads the chat's
    record, which they can't, else "other"."""
    if namespace == target or namespace.startswith(f"{target}."):
        return "here"
    if reads_record(registry, target):
        return "record"
    return "other"


def reads_record(registry, namespace):
    if namespace not in registry.names:
        return namespace == HISTORIAN
    return sees(registry, namespace, HISTORY_TOOL)


def sees(registry, namespace, tool):
    """Whether namespace *namespace*'s agents see *tool*, granted there or above."""
    return tool in registry.resolve(namespace).tools


def impossible_hand_offs(registry, namespace, targets):
    """Each worked example *namespace* is shown that hands off where its agents can't,
    with where to and why not; *targets* as hand_off_targets() gives them."""
    resolved = registry.resolve(namespace)
    found = []
    for example in resolved.decompositions:
        level = resolved.sources.decompositions[example.name]
        target = targets.get((level, example.name))
        why = target and unreachable(registry, namespace, target)
        if why:
            found.append((example, target, why))
    return found


def saying_so(registry, namespace, example, target, why):
    """*example* with its hand-off, which *namespace* can't make, replaced by what its
    agents do instead."""
    kind = hand_off_kind(registry, namespace, target)
    answer = repr(CANT_ANSWER.render(target=target, why=why))
    record = reads_record(registry, target)
    reply = CANT_HAND_OFF.render(
        kind=kind, target=target, why=why, answer=answer, record=record
    )
    messages = example.model_dump()["messages"][:-1]
    return {
        "name": example.name,
        "messages": [*messages, {"role": "assistant", "content": reply}],
    }


def without_impossible_hand_offs(config, run, registry):
    """*config* with each worked example that hands off where its namespace can't (one its
    spawn list leaves out, or one there is no longer) replaced, in that namespace, by what
    its agents do instead: copying the hand-off, the agent was refused, then handed the
    question to another namespace, which inherits the same example, until the depth ran
    out. A namespace holds an example only where what it inherits differs from what it
    should see: the replacement, or the original back below one that can't. *run* and
    *registry* are *config* as run_config_and_registry() gives them."""
    targets = hand_off_targets(run)
    swaps = {
        name: {
            example.name: saying_so(registry, name, example, target, why)
            for example, target, why in impossible_hand_offs(registry, name, targets)
        }
        for name in registry.names
    }
    swapped = {example for each in swaps.values() for example in each}
    shown = {}
    for name in in_chain_order(registry):
        chain = registry.chain(name)
        above = shown[chain[-2]] if len(chain) > 1 else {}
        shown[name] = {
            example.name: swaps[name].get(example.name) or example.model_dump()
            for example in registry.resolve(name).decompositions
            if example.name in swapped
        }
        fields = config["namespaces"].get(name)
        if fields is not None:
            own = {each["name"]: each for each in fields["decompositions"]}
            own |= overrides(own, above, shown[name], swaps[name])
            fields["decompositions"] = list(own.values())
    return config


def overrides(own, above, shown, swaps):
    """What a namespace holding the examples *own* must hold itself to show its agents
    *shown*, when the namespace above it shows *above*: each one it replaces, and each
    other one it doesn't hold that differs from above's."""
    return {
        name: example
        for name, example in shown.items()
        if (name in swaps if name in own else above.get(name) != example)
    }


def engine_config(library_path, namespace, tools_dir, chat, run, key_names=()):
    """The Library as one plain dict that V2Config accepts, for run *run* of *chat*, which
    enters *namespace*; the run's model and client are the chat's to put in.

    Only the tools some namespace grants are in it, the app's own included: a run builds
    each tool it declares, whichever namespace it enters, so a tool any namespace grants
    runs its module and factory at the start of every run of every chat, one no namespace
    grants never runs, and one that fails to build stops every run until no namespace
    grants it. Each Library tool's source is written into
    *tools_dir*, and its factory_from points at that file. Each of the app's tools is told
    the app file, the chat, the run, *tools_dir*, the namespace it enters, and *key_names*,
    the variables of the worker's environment that hold keys; the library tool also the
    namespaces whose chats may write with it (editors).
    """
    tools_dir = Path(tools_dir).resolve()
    tools_dir.mkdir(parents=True, exist_ok=True)
    with closing(open_library(library_path)) as connection:
        read = read_library(connection)
    config = config_of(read)
    grants = granted(config["namespaces"])
    config["tools"] = {
        name: block for name, block in config["tools"].items() if name in grants
    }
    for name, (_, source) in read["tools"].items():
        if source is None or name not in config["tools"]:
            continue
        file = tools_dir / tool_file(name)
        file.write_text(source, encoding="utf-8")
        config["tools"][name]["factory_from"] = str(file)
    told = {
        "library": str(Path(library_path).resolve()),
        "chat": chat,
        "run": run,
        "tools_dir": str(tools_dir),
        "namespace": namespace,
        "key_names": list(key_names),
    }
    for name in APP_TOOLS.keys() & config["tools"]:
        config["tools"][name] |= told
    try:
        resolved, registry = run_config_and_registry(config)
        editors = [
            name for name in registry.names if sees(registry, name, LIBRARY_TOOL)
        ]
        config = without_impossible_hand_offs(config, resolved, registry)
        config = with_run_lines(config, registry)
    except ValueError as error:
        # The worker refuses such a Library itself, in its own words.
        log.warning("library.not_checked", error=str(error))
        editors = []
    if LIBRARY_TOOL in config["tools"]:
        config["tools"][LIBRARY_TOOL]["editors"] = editors
    return config | {"entry_namespace": namespace}


def with_run_lines(config, registry):
    """*config* with each of run_lines() after its namespace's own system_suffix, a
    blank line between, for its run only: the Library holds none."""
    for name, line in run_lines(registry).items():
        fields = config["namespaces"][name]
        own = (fields.get("system_suffix") or "").rstrip("\n")
        fields["system_suffix"] = "\n\n".join(filter(None, [own, line]))
    return config


def default_namespace(library_path):
    """Where a new chat starts: the run settings' entry_namespace, root unless one is set."""
    with closing(open_library(library_path)) as connection:
        return stored_settings(connection).get(ENTRY_NAMESPACE, ROOT)


def has_namespace(connection, name):
    found = connection.execute("SELECT 1 FROM namespaces WHERE name = ?", (name,))
    return found.fetchone() is not None


def namespace_yaml(connection, name):
    """Namespace *name*'s own YAML, as a change starts from it and logs it as its before:
    for one the engine can't read, as a shared file may hold, a note saying why; None
    when the Library has no namespace *name*."""
    _, left, shown = read_namespaces(connection)
    return left_out_note(left[name]) if name in left else shown.get(name)


def check_namespace(connection, name):
    """Namespace *name*'s own YAML; raise if the Library has no namespace *name*."""
    found = namespace_yaml(connection, name)
    if found is None:
        raise LibraryError(
            f'The Library has no namespace "{name}"; pick one from the list.'
        )
    return found


def config_and_registry(connection):
    """The Library as a run's config, with the llm tool a model brings, and its namespaces
    as the engine resolves them."""
    return run_config_and_registry(stored_config(connection))


def run_config_and_registry(config):
    """*config*, a plain dict in the engine's shape, as a run's config with the llm tool a
    model brings, and its namespaces as the engine resolves them."""
    run = V2Config.model_validate(config | A_MODEL)
    return run, build_namespace_registry(run)


def check(connection):
    """Raise unless the engine would accept the Library: every chain resolves and every
    tool granted exists. The Library as a run's config, and its namespaces as the engine
    resolves them."""
    config, registry = config_and_registry(connection)
    registry.check(declared_tool_names(config))
    return config, registry


def check_backbones(registry, names):
    """Raise unless each of the namespaces *names* runs on a backbone the engine builds."""
    for name in names:
        kind = registry.backbone(name)
        if kind not in BACKBONES:
            level = registry.resolve(name).sources.reasoner or name
            known = ", ".join(map(repr, BACKBONES))
            raise LibraryError(
                f"The engine refuses it: namespace {level!r}: unknown reasoner type"
                f" {kind!r}; known: {known}."
            )


def held_in(value):
    """What *value* holds, keys included, when it is a mapping, list, or set or the pairs
    !!set, !!omap and !!pairs make; None for any other value."""
    if isinstance(value, dict):
        return [*value, *value.values()]
    if isinstance(value, list | tuple | set | frozenset):
        return list(value)
    return None


def texts_in(data):
    """Every text in parsed YAML *data*, keys included, and bytes read as latin-1, as a
    key written as !!binary is; each mapping or list once, as aliases may share or nest them."""
    stack, seen = [data], set()
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            yield value
        elif isinstance(value, bytes):
            yield value.decode("latin-1")
        elif (held := held_in(value)) is not None and id(value) not in seen:
            seen.add(id(value))
            stack.extend(held)


def expanded_size(data, cap):
    """How big parsed YAML *data* is with each alias written out in full, as the engine
    expands them, as values plus their text's characters; past *cap* for one that holds
    itself, which never ends. Each mapping or list is measured once, so it takes as long
    as the text did to parse."""
    if held_in(data) is None:
        return scalar_size(data)
    sizes, stack = {}, [data]
    while stack:
        value = stack[-1]
        held = held_in(value)
        inner = [each for each in held if held_in(each) is not None]
        if id(value) not in sizes:
            # None while what it holds is measured: one of those holding it is a loop.
            sizes[id(value)] = None
            if any(sizes.get(id(each), 0) is None for each in inner):
                return cap + 1
            stack += [each for each in inner if id(each) not in sizes]
            continue
        stack.pop()
        if sizes[id(value)] is None:
            sizes[id(value)] = 1 + sum(
                scalar_size(each) if held_in(each) is None else sizes[id(each)]
                for each in held
            )
    return sizes[id(data)]


def scalar_size(value):
    return 1 + len(value) if isinstance(value, str | bytes) else 1


def literals(tree):
    """Each text and bytes literal in the parsed Python *tree*, as an escape can spell a
    key the raw text doesn't hold."""
    constants = [
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
    ]
    return list(texts_in(constants))


def refuse_keys(texts, secrets, refusal):
    """Raise LibraryError(*refusal*) if any of *texts* holds one of *secrets*."""
    if any(secret in text for secret in secrets for text in texts):
        raise LibraryError(refusal)


def taken(name):
    """Whether a tool of your own may not have *name*: a Python keyword, a built-in's or
    the app's, or one the REPL binds itself."""
    return keyword.iskeyword(name) or name in TAKEN_NAMES


def problem(error):
    """One of the Decomposition model's validation errors, as the panel words it."""
    match error["loc"], error["type"]:
        case (key,), "extra_forbidden":
            return f'Remove "{key}": only name and messages go in the YAML; pick the namespace below.'
        case ("name",), "missing":
            return 'Add a name: line, like "name: count the open tickets".'
        case ("name",), _:
            return 'The name should be text, like "name: count the open tickets".'
        case ("messages",), "missing":
            return "Add messages: a list of turns, each with a role and content, as in the template."
        case ("messages",), _:
            return 'messages should be a list of at least one turn, each starting "- role:".'
        case ("messages", int(index)), _:
            return f"Message {index + 1} should be a mapping with role and content."
        case ("messages", int(index), key), "extra_forbidden":
            return f'Message {index + 1}: remove "{key}"; a message holds only role and content.'
        case ("messages", int(index), "role"), "missing":
            return f"Message {index + 1}: role should be {ROLES}."
        case ("messages", int(index), "role"), _:
            given = json.dumps(error["input"], default=str)
            return f"Message {index + 1}: role should be {ROLES}, not {given}."
        case ("messages", int(index), "content"), "missing":
            return f"Message {index + 1} needs content: the turn's text."
        case ("messages", int(index), "content"), _:
            return f'Message {index + 1}: content should be text; quote it, or write "content: |" with the text indented below.'
    return f"{'.'.join(map(str, error['loc']))}: {error['msg']}."


def loaded(text):
    """The YAML *text* as data, or a LibraryError saying where it doesn't parse or why it
    holds what no text can."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f"line {mark.line + 1}: " if mark else ""
        reason = getattr(error, "problem", None) or error
        raise LibraryError(
            f"The YAML doesn't parse ({where}{reason}). Check the indentation and quoting on that line."
        ) from error
    except RecursionError as error:
        raise LibraryError("The YAML is nested too deeply to read.") from error
    check_aliases(data, text)
    half = next(filter(None, map(SURROGATE.search, texts_in(data))), None)
    if half:
        raise LibraryError(
            f"The YAML escapes half of a character (U+{ord(half[0]):04X}), which no text"
            " can hold; write the whole character instead."
        )
    return data


def check_no_run_lines(data):
    """Raise if a text in parsed YAML *data* reads as one of the lines a run adds after a
    namespace's system_suffix (RUN_LINE_SHAPES), however spaced, cased or split: agents
    would take it for the app's."""
    for text in texts_in(data):
        if found := RUN_LINE_SHAPES.search(plain_words(text)):
            raise LibraryError(
                f"It holds “{found.group()}”, which reads as a line the app adds after a"
                " namespace's system_suffix at run time (which namespace its agents are"
                " in, where their Python runs, where they may hand work), so agents would"
                " take it for the app's: take it out."
            )


def plain_words(text):
    """*text* as its words read: in its compatible forms (NFKC, so a non-breaking or
    full-width space is a space), without the characters that don't show, apostrophes
    straight, each run of spaces and line breaks one space, case folded."""
    text = unicodedata.normalize("NFKC", text).translate(unseen_and_curly())
    return " ".join(text.split()).casefold()


@cache
def unseen_and_curly():
    """What plain_words() writes each character that doesn't show, but a space, and each
    curly apostrophe, as: none, and a straight one."""
    unseen = [char for char in map(chr, range(sys.maxunicode + 1)) if invisible(char)]
    table = {ord(char): None for char in unseen if not char.isspace()}
    return table | {ord(char): "'" for char in CURLY_APOSTROPHES}


def check_aliases(data, text):
    """Raise if the YAML *text*, parsed as *data*, stands for much more than it holds once
    each alias (*name) is written out, as the engine writes them out: a few lines of them
    can stand for millions of values."""
    allowed = ALIASES_STAND_FOR * len(text) + ALIASES_ALLOWANCE
    if expanded_size(data, allowed) > allowed:
        raise LibraryError(
            "Its aliases (*name) stand for too much: written out, as the engine reads"
            f" them, it would hold more than {allowed:,} values and characters, from"
            f" {len(text):,} characters of YAML. Write it without aliases."
        )


def fields_in(text):
    """The fields a namespace's YAML *text* sets, none when it is blank; a LibraryError
    when it isn't a mapping of them."""
    data = loaded(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise LibraryError(FIELDS_MAPPING)
    return data


def decomposition_in(text):
    """The decomposition the YAML *text* describes, as the engine's own model reads it."""
    data = loaded(text)
    if not isinstance(data, dict):
        raise LibraryError(
            "The YAML should be a mapping: a name: line, then messages:, as in the template."
        )
    try:
        return Decomposition.model_validate(data)
    except ValidationError as error:
        raise LibraryError("\n".join(map(problem, error.errors()))) from error


def parsed(text):
    """The decomposition the YAML *text* describes, checked by the engine's own model, as
    a save takes one (check_saved())."""
    decomposition = decomposition_in(text)
    check_saved(decomposition)
    return decomposition


def check_saved(decomposition):
    """Raise unless *decomposition* is one a save takes: named as the panel names one, and
    holding no line the app adds at run time."""
    check_decomposition_name(decomposition.name)
    check_no_run_lines(decomposition.model_dump())


def check_decomposition_name(name):
    """Raise unless a decomposition's *name* reads as it looks: not blank, with no outer
    space, character you can't see, or word mixing alphabets whose letters pass for one
    another's."""
    if not name.strip():
        raise LibraryError(BLANK_NAME)
    if name != name.strip():
        raise LibraryError(
            f'The name "{name}" starts or ends with a space; take it out, so it reads as it'
            " looks."
        )
    hidden = first_char(name, invisible)
    if hidden:
        raise LibraryError(
            f"The name holds a character you can't see ({hidden}); take it out, so it"
            " reads as it looks."
        )
    for word in re.findall(r"\w+", name):
        scripts = Counter(map(script_of, filter(str.isalpha, word)))
        if len(scripts.keys() & LOOK_ALIKE_SCRIPTS) > 1:
            main = scripts.most_common(1)[0][0]
            odd = next(
                each for each in word if each.isalpha() and script_of(each) != main
            )
            raise LibraryError(
                f"The name mixes alphabets in the word {word!a} (U+{ord(odd):04X}), so it"
                " could pass for another; write each word in one alphabet, so it reads as"
                " it looks."
            )


def script_of(letter):
    """The alphabet *letter* is of, as its Unicode name begins: LATIN, CYRILLIC, ..."""
    return unicodedata.name(letter, "").partition(" ")[0]


def first_char(text, test):
    """The first character of *text* that passes *test*, as U+ writes it; None for none."""
    return next((f"U+{ord(char):04X}" for char in text if test(char)), None)


def invisible(char):
    return unicodedata.category(char) in INVISIBLE


# Root's worked examples in a new Library, as YAML: its hand-offs to the librarian, then
# to the historian, last as nearer the task they are followed more.
ROOTS_EXAMPLES = [*LIBRARIAN_HAND_OFFS, *map(hand_off, HISTORIANS_EXAMPLES)]


def check_size(text, what):
    """Raise unless *text*, *what* the Library would hold, fits in a row: at most
    MAX_YAML_CHARS characters, as its readers count once a row is loaded."""
    if len(text) > MAX_YAML_CHARS:
        raise LibraryError(
            f"{what} may be at most {MAX_YAML_CHARS:,} characters; this has"
            f" {len(text):,}."
        )


def sandbox_templates():
    """Make every Jinja render the engine does in this process sandboxed: each process
    that runs the engine calls it first."""
    deep_reasoner.llm._JinjaEnv = SANDBOXED


def template_names(variables):
    """The names a turn's template may read: what the engine fills in, and the run
    settings' prompt_template_variables, *variables*."""
    return frozenset({*FILLED_IN, *variables})


def first_use(template, name):
    """The line of *template* (parsed) where *name* is first read."""
    uses = [
        each.lineno
        for each in template.find_all(nodes.Name)
        if each.name == name and each.ctx == "load"
    ]
    return min(uses)


def internal_read(node):
    """The Python internal *node* reads by a name it gives, which a run's sandbox
    refuses; None for none."""
    if isinstance(node, nodes.Getattr):
        name = node.attr
        return name if name.startswith("_") or name in UNSAFE_NAMES else None
    if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        key = node.arg.value
        return key if isinstance(key, str) and key.startswith("_") else None
    return None


def refused(node):
    """Why a run can't render a template holding *node*, in words; None when it can."""
    if isinstance(node, INCLUDES):
        return "includes another template, which a run can't"
    if internal := internal_read(node):
        return f"reaches into Python internals ({internal}), which a run refuses"
    return None


def first_refused(template):
    """The line of *template* (parsed) where it first holds what a run can't render, and
    why; None when it holds none."""
    found = template.find_all(SCREENED)
    return next(((n.lineno, why) for n in found if (why := refused(n))), None)


def template_problem(label, text, names):
    """Why the engine couldn't render the turn *text*, *label*, when it includes another
    template, reads a Python internal, reads more than *names* or doesn't parse or
    compile; None when it holds no template or none of these. It is never rendered, so
    this runs none of its code."""
    if not _has_jinja(text):
        return None
    try:
        template = JINJA.parse(text)
        if found := first_refused(template):
            line, why = found
            return f"{label}, line {line}: {TEMPLATE}, and this one {why}. {ESCAPE}"
        missing = sorted(meta.find_undeclared_variables(template) - names)
        if missing:
            line = first_use(template, missing[0])
            return f'{label}, line {line}: {TEMPLATE}, and this one names "{missing[0]}"; {NAMES_ONLY}. {ESCAPE}'
        JINJA.compile(template)
    except TemplateSyntaxError as error:
        return f"{label}, line {error.lineno}: {TEMPLATE}, and this one doesn't parse ({error.message}). {ESCAPE}"
    except TEMPLATE_FAILURES as error:
        # A SyntaxError's own line is in the Python Jinja generates, not in the turn.
        reason = getattr(error, "msg", None) or str(error) or type(error).__name__
        return (
            f"{label}: {TEMPLATE}, and this one can't be compiled ({reason}). {ESCAPE}"
        )
    return None


def template_problems(decomposition, names):
    """Each turn of *decomposition* the engine couldn't render, its template reading
    only *names*."""
    turns = enumerate(decomposition.messages, start=1)
    found = (
        template_problem(f"Turn {number} ({turn.role})", turn.content, names)
        for number, turn in turns
    )
    return [each for each in found if each]


def check_templates(connection, decomposition):
    """Raise unless every turn of *decomposition* is a template the engine renders."""
    config, _ = config_and_registry(connection)
    names = template_names(config.prompt_template_variables)
    problems = template_problems(decomposition, names)
    if problems:
        raise LibraryError("\n".join(problems))


def running_code(content):
    """The code the engine runs from the assistant turn *content*, blank for none."""
    try:
        return code(content).source
    except NoCodeBlock:
        return ""


def answers(content):
    """Whether the code the engine runs from *content* calls FinalAnswer, outside a
    whole-line comment."""
    lines = running_code(content).split("\n")
    return any(
        FINAL_ANSWER.search(line) for line in lines if not COMMENT_LINE.match(line)
    )


def reply_problems(label, content):
    """What the assistant turn *label* lacks: its reasoning, and the one block the engine runs."""
    blocks = content.count(REPL)
    problems = [] if THINK.search(content) else [f"{label} has no <think>…</think>."]
    if blocks == 0:
        problems.append(f"{label} has no <repl>…</repl> block.")
    if blocks > 1:
        problems.append(
            f"{label} has {blocks} <repl> blocks; the engine runs only the last, so keep one."
        )
    if blocks and not running_code(content):
        problems.append(
            f"{label} has an empty <repl> block; the engine treats it as no block."
        )
    return problems


def problems_of_turn(turn, previous):
    """The problems of *turn*, a (number, role, content), after *previous*, the latest turn
    before it but a system one (None for none)."""
    number, role, content = turn
    label = f"Turn {number} ({role})"
    if role == "system" and previous is None:
        return []
    if role == "system":
        return [f"{label} comes after the task; system turns go before it."]
    before, before_role, before_content = previous or (None, None, "")
    problems = []
    if before is None and role == "assistant":
        problems.append(f"{label} should be the user's task; start with a user turn.")
    if before_role == role:
        problems.append(f"{label} follows another {role} turn; {BETWEEN[role]}.")
    if role == "assistant":
        return problems + reply_problems(label, content)
    # What a cell printed follows it, unless it answered: then a new message may.
    if (
        before_role == "assistant"
        and not answers(before_content)
        and not OBSERVATION.search(content)
    ):
        problems.append(
            f"{label} should hold the <observation>…</observation> that turn {before}'s code printed."
        )
    return problems


def turn_problems(messages):
    """What keeps *messages*, a decomposition's turns (each a mapping of role and content),
    from being a worked example like deep_reasoner's own, in plain words; none when it is one.
    Its code isn't run, so an observation isn't checked against what the code would print."""
    problems = []
    previous = None
    for number, message in enumerate(messages, start=1):
        turn = (number, message["role"], message["content"])
        problems += problems_of_turn(turn, previous)
        if message["role"] != "system":
            previous = turn
    if previous is None:
        return [NO_TASK]
    number, role, content = previous
    if role == "user":
        problems.append(
            f"It ends on turn {number} (user); end with the assistant's turn that calls FinalAnswer(...)."
        )
    elif not answers(content):
        problems.append(
            f"Turn {number} (assistant), the last, should call FinalAnswer(...) in its <repl>."
        )
    return problems


def example_problems(decomposition):
    """The problems of *decomposition* as a worked example: a shared file's may have the
    blank name parsed() refuses."""
    if not decomposition.name.strip():
        return [BLANK_NAME]
    return turn_problems(decomposition.model_dump()["messages"])


def decomposition_problems(decomposition, names):
    """The problems of *decomposition*, as a worked example and as templates reading only
    *names*."""
    return example_problems(decomposition) + template_problems(decomposition, names)


def add_decomposition(connection, namespace, text, replace=False, allowed=None):
    """Store the decomposition written as YAML *text* under *namespace*, or raise
    LibraryError saying why not: one of its name there is refused, or with *replace* put in
    its place. *allowed*, given the connection under the write lock first, raises to refuse.
    A worked example with problems is stored all the same, so a draft can be saved.
    The stored row with its problems, and the YAML it replaced (None when it is new)."""
    check_size(text, "A decomposition's YAML")
    decomposition = parsed(text)
    name = decomposition.name
    stored = {"namespace": namespace, "name": name, "yaml": text}
    with writing(connection):
        if allowed is not None:
            allowed(connection)
        # Under the lock, so a namespace deleted meanwhile leaves no decomposition behind.
        check_namespace(connection, namespace)
        found = stored_decomposition(connection, namespace, name)
        if found is not None and not replace:
            raise already_has(namespace, name)
        check_templates(connection, decomposition)
        connection.execute(UPSERT_DECOMPOSITION, tuple(stored.values()))
    problems = example_problems(decomposition)
    return stored | {"problems": problems}, found


def decomposition_row(connection, namespace, name):
    """*namespace*'s decomposition *name* as fitting() reads it, (text, size); None when it
    has none."""
    return connection.execute(
        f"SELECT {fitting('yaml')} FROM decompositions WHERE namespace = ? AND name = ?",
        (namespace, name),
    ).fetchone()


def stored_decomposition(connection, namespace, name):
    """*namespace*'s decomposition *name*, as a change logs it as its before: its YAML, or
    for one the engine can't read, as a shared file may hold, a note saying why; None
    when it has none."""
    found = decomposition_row(connection, namespace, name)
    if found is None:
        return None
    try:
        text = row_text(*found)
        decomposition_in(text)
    except LibraryError as error:
        return left_out_note(str(error))
    return text


def already_has(namespace, name):
    return LibraryError(
        f'{namespace} already has "{name}"; rename it, pick another namespace, or Edit'
        " that one."
    )


def edit_decomposition(connection, namespace, name, was, text):
    """Put the decomposition written as YAML *text* in the place of *namespace*'s
    decomposition *name*, which the person opened as *was*; it may take another name
    that *namespace* doesn't hold. Raises LibraryError saying why not, and
    ChangedMeanwhile if *name* is no longer stored as *was*. The name it is stored
    under."""
    check_size(text, "A decomposition's YAML")
    decomposition = parsed(text)
    with writing(connection):
        check_templates(connection, decomposition)
        # In its place, so the decompositions stay in the order they were added.
        try:
            edited = connection.execute(
                "UPDATE decompositions SET name = ?, yaml = ?"
                " WHERE namespace = ? AND name = ? AND yaml = ?",
                (decomposition.name, text, namespace, name, was),
            )
        except sqlite3.IntegrityError as error:
            raise already_has(namespace, decomposition.name) from error
        if edited.rowcount == 0:
            raise changed_meanwhile(f'"{name}" in {namespace}')
    return decomposition.name


def delete_decomposition(connection, namespace, name, was):
    """Delete *namespace*'s decomposition *name*, which the person was shown as *was*, or
    as None for one too long to show; ChangedMeanwhile if it is no longer stored so, or at
    all."""
    with writing(connection):
        found = decomposition_row(connection, namespace, name)
        if found is None:
            raise ChangedMeanwhile(
                f'"{name}" in {namespace} was already deleted or renamed {BY_WHOM}.'
            )
        if shown_text(*found) != was:
            raise ChangedMeanwhile(
                f'"{name}" in {namespace} was changed after you opened it {BY_WHOM}, so'
                " it wasn't deleted. Look at it as it is now, then delete it again if"
                " you still want to."
            )
        connection.execute(
            "DELETE FROM decompositions WHERE namespace = ? AND name = ?",
            (namespace, name),
        )


def namespace_problem(error):
    """One of the NamespaceConfig model's validation errors, as the panel words it."""
    match error["loc"], error["type"]:
        case (key,), "extra_forbidden":
            return f'Remove "{key}": a namespace\'s YAML holds only {NAMESPACE_HOLDS}.'
        case ("vars", name, "[key]"), _:
            return f'A var\'s name should be text, like "year", not {name}.'
        case (str(key), *_), _ if key in SHAPES:
            return SHAPES[key]
    return f"{'.'.join(map(str, error['loc']))}: {error['msg']}."


def authored(name, text):
    """Namespace *name* as its YAML *text* describes it, checked by the engine's own model,
    and the YAML to store for it."""
    check_size(text, "A namespace's YAML")
    data = fields_in(text)
    problems = [NAMED_ELSEWHERE[key] for key in NAMED_ELSEWHERE if key in data]
    fields = {key: value for key, value in data.items() if key not in NAMED_ELSEWHERE}
    try:
        namespace = NamespaceConfig.model_validate(fields | {"name": name})
    except ValidationError as error:
        problems += map(namespace_problem, error.errors())
        raise LibraryError("\n".join(problems)) from error
    if problems:
        raise LibraryError("\n".join(problems))
    check_no_run_lines(fields)
    # As typed, comments and all, but for the one text an untouched root is kept as.
    return namespace, own_yaml(text)


def own_yaml(text):
    """A namespace's stored YAML *text* as the panel shows and saves it: blank for one
    that sets nothing."""
    return "" if text == EMPTY_NAMESPACE else text


def accepted(connection):
    """Raise unless the engine accepts the Library, in words for the panel; check's answer."""
    try:
        return check(connection)
    except ValueError as error:
        raise LibraryError(f"The engine refuses it: {error}") from error


def var_problem(name):
    """Why *name* won't do as a var's name, None when it will: the agent's code reads a
    var by its name."""
    if not name.isidentifier():
        return (
            f'Var "{name}" needs a Python name, like "year": letters, digits and _,'
            " not starting with a digit."
        )
    if keyword.iskeyword(name):
        return f'Var "{name}" is a Python keyword; rename it, like "{name}_value".'
    if name.startswith("_"):
        plain = name.lstrip("_") or "value"
        return f'Var "{name}" starts with _, which templates may not read; rename it, like "{plain}".'
    if name in BUILTIN_NAMES:
        return (
            f"Var \"{name}\" would hide Python's own {name}, which the agent's code may"
            f' use; rename it, like "{name}_value".'
        )
    return None


def tool_name_problem(name):
    """Why a tool of your own may not be named *name*, beside taken(): None when it may.
    As a var, it would hide one of Python's built-ins from the agent's code; or a module
    that code commonly uses."""
    if name in BUILTIN_NAMES:
        hidden = f"Python's own {name}"
    elif name in COMMON_MODULES:
        hidden = f"the {name} module"
    else:
        return None
    return (
        f'Tool "{name}" would hide {hidden}, which the agent\'s code may use; rename it,'
        f' like "{name}_tool".'
    )


def hidden_vars(registry, names):
    """Raise if, in any of the namespaces *names*, a tool takes the place of a var of
    the same name, as the engine binds tools over vars."""
    for each in names:
        resolved = registry.resolve(each)
        clashes = sorted(resolved.vars.keys() & set(resolved.tools))
        if clashes:
            clash = clashes[0]
            raise LibraryError(
                f'In {each}, var "{clash}" (set in {resolved.sources.vars[clash]}) has'
                f" the name of the tool {clash} (granted in"
                f" {resolved.sources.tools[clash]}), which takes its place: rename the var."
            )


def suffix_problem(connection, namespace):
    """Why the engine couldn't render *namespace*'s system_suffix, as template_problem
    words it; None when it can."""
    suffix = namespace.system_suffix or ""
    if not _has_jinja(suffix):
        return None
    config, _ = config_and_registry(connection)
    label = f"The system_suffix of {namespace.name}"
    return template_problem(
        label, suffix, template_names(config.prompt_template_variables)
    )


def check_fields(connection, namespace, adding):
    """Raise unless *namespace*'s vars are well named, its system_suffix renders, and
    every namespace it may spawn into is in the Library, or is itself; *adding*, as when
    it is new."""
    problems = [
        *map(var_problem, namespace.vars),
        suffix_problem(connection, namespace),
    ]
    problems = [each for each in problems if each]
    if problems:
        raise LibraryError("\n".join(problems))
    # The engine lets a spawn name no namespace, and fails only when the agent spawns there.
    name = namespace.name
    missing = [
        spawned
        for spawned in namespace.spawn or []
        if spawned != name and not has_namespace(connection, spawned)
    ]
    if missing:
        raise LibraryError(
            "\n".join(missing_spawn(spawned, name, adding) for spawned in missing)
        )


def missing_spawn(spawned, name, adding):
    """Why namespace *name*'s spawn can't name *spawned*, which the Library doesn't hold,
    and what to do; *adding*, as when *name* is new."""
    if adding and spawned.startswith(f"{name}."):
        return (
            f'spawn names "{spawned}", which goes under {name} and isn\'t in the'
            f" Library yet: add {name} without it first, then {spawned}, then put"
            f" {spawned} in the spawn of {name}."
        )
    return (
        f'spawn names "{spawned}", which is not a namespace in the Library: add'
        f" {spawned} first, or take it out of spawn."
    )


def write_namespace(connection, name, text, adding):
    """Store namespace *name* as its YAML *text*, inside the caller's transaction, which
    keeps it only if it passes check_written; *adding*, as when it is new."""
    connection.execute(UPSERT_NAMESPACE, (name, text))
    check_written(connection, name, adding)


def check_written(connection, name, adding):
    """Raise unless namespace *name*, as just written (new, with *adding*), is one the
    panel saves: the engine accepts the Library and reads it; its fields pass
    check_fields; and there and below, each backbone is one the engine builds and no tool
    takes a var's place. A namespace elsewhere is not its to answer for. The panel's
    saves and an import each call it, so a check added here reaches both."""
    _, registry = accepted(connection)
    # Root's too, which the engine supplies bare when the file's is left out.
    left = read_namespaces(connection)[1]
    if name in left:
        raise LibraryError(left[name])
    check_fields(connection, registry.config(name), adding)
    below = [each for each in registry.names if name in registry.chain(each)]
    check_backbones(registry, below)
    hidden_vars(registry, below)


def add_namespace(connection, name, text, allowed=None):
    """Store a new namespace *name* with the fields its YAML *text* gives, or raise
    LibraryError saying why not; the YAML stored. Its decompositions are added afterwards.
    *allowed*, given the connection under the write lock first and the namespace, raises to
    refuse."""
    check_name(name)
    namespace, stored = authored(name, text)
    with writing(connection):
        if allowed is not None:
            allowed(connection, namespace)
        check_place(connection, name)
        write_namespace(connection, name, stored, adding=True)
    return stored


def check_name(name):
    """Raise unless *name* names a new namespace: plain words, joined by dots for a
    child, none of them root by another spelling, as root is every namespace's parent."""
    if not name:
        raise LibraryError(NAME_IT)
    if odd := first_char(name, not_plain):
        raise LibraryError(
            f"The name holds a character a namespace's name can't ({odd}), so it could"
            f" pass for another namespace in the menu. {NAME_SHAPE}"
        )
    if not NAMESPACE_NAME.fullmatch(name):
        raise LibraryError(NAME_SHAPE)
    if reads_as_root(name):
        raise LibraryError(
            f'"{name}" reads as root, which every namespace goes under already: name it'
            ' something else, like "research".'
        )


def not_plain(char):
    return not (char.isascii() and char.isprintable())


def reads_as_root(name):
    return ROOT in [part.strip().lower() for part in name.split(".")]


def shown_name(name):
    """A namespace's *name* as the app writes it: as it is in printable ASCII, else as
    ascii() writes it, as check_stored_name() does, so a look-alike never passes for
    another."""
    return name if name.isascii() and name.isprintable() else ascii(name)


def check_stored_name(name):
    """Raise unless a stored namespace's *name*, root's aside, is one a save takes
    (check_name()), written out so the panel shows what it holds: one that isn't could
    pass for another in the menu, as "historian " for historian."""
    if name == ROOT:
        return
    passes = "so it could pass for another namespace in the menu"
    if odd := first_char(name, not_plain):
        why = f"holds characters a saved name can't ({odd}), {passes}"
    elif reads_as_root(name):
        why = f"reads as root, {passes}"
    elif not NAMESPACE_NAME.fullmatch(name):
        why = f"isn't one a save takes: {NAME_SHAPE[0].lower()}{NAME_SHAPE[1:-1]}"
    else:
        return
    raise LibraryError(f"Its name, {name!a}, {why}.")


def check_place(connection, name):
    """Raise unless a new namespace *name* has a place in the Library: none holds its
    name, and the one it goes under is there."""
    if has_namespace(connection, name):
        raise LibraryError(
            f'There is already a namespace "{name}"; open it and use Edit YAML.'
        )
    parent = name.rpartition(".")[0]
    if parent and not has_namespace(connection, parent):
        raise LibraryError(f'Add namespace "{parent}" first: "{name}" goes under it.')


def fields_of(text):
    """The fields a namespace's own YAML *text* sets: none when it sets none, or isn't
    YAML, as a change a cell logged may not be."""
    try:
        fields = loaded(text) if type(text) is str else None
    except LibraryError:
        return {}
    return fields if isinstance(fields, dict) else {}


def merged(before, text):
    """A namespace's own YAML *before* with each field the YAML *text* sets in place of
    its own, and each it sets to null taken out; the rest kept, as typed when none
    changes."""
    given = fields_in(text)
    own = fields_of(before)
    fields = dict(own)
    for key, value in given.items():
        if value is None:
            fields.pop(key, None)
        else:
            fields[key] = value
    if fields == own:
        return before
    together = dump(fields) if fields else ""
    check_size(together, "With this change, the namespace's YAML")
    return together


def change_namespace(connection, name, text, allowed=None, was=NOT_OPENED, merge=False):
    """Give namespace *name* the fields its YAML *text* gives in place of its own, keeping
    its decompositions, or raise LibraryError saying why not; with *merge*, only those
    fields, keeping the rest. *allowed* as add_namespace takes it. Given *was*, the YAML
    the person opened it on, None for one too long to show, ChangedMeanwhile unless it is
    still stored so. The YAML it replaced (namespace_yaml()), and the YAML stored."""
    with writing(connection):
        if merge:
            text = merged(namespace_yaml(connection, name) or "", text)
        namespace, stored = authored(name, text)
        if allowed is not None:
            allowed(connection, namespace)
        before = check_unchanged(connection, name, was)
        write_namespace(connection, name, stored, adding=False)
    return before, stored


def check_unchanged(connection, name, was):
    """Namespace *name*'s own YAML, as namespace_yaml() gives it; raise if the Library
    has none of that name, or, given *was*, the YAML the person opened it on,
    ChangedMeanwhile unless it is still stored so."""
    before = check_namespace(connection, name)
    shown = read_namespaces(connection)[2][name]
    if was is not NOT_OPENED and own_yaml(shown) != was:
        raise changed_meanwhile(f"Namespace {name}")
    return before


def granted_afresh(connection, namespace):
    """The custom tools *namespace* grants that no namespace of the Library grants yet,
    so no run builds yet, each as its stored block and source. A Library the engine
    can't resolve, as a shared file may hold, grants none, so a save that mends it can be
    made, its tools built first."""
    try:
        _, registry = config_and_registry(connection)
    except ValueError:
        granted = set()
    else:
        granted = {
            tool for each in registry.names for tool in registry.config(each).tools
        }
    afresh = set(namespace.tools) - granted
    return {
        name: (block, source)
        for name, (block, source) in library_tools(connection).items()
        if name in afresh and source is not None
    }


def built_grants(
    connection, name, text, build, adding, merge=False, was=NOT_OPENED, locks=None
):
    """The custom tools namespace *name*, as the YAML *text* makes it (merged into its
    own with *merge*), grants afresh (granted_afresh()), each built by *build* once every
    check that costs nothing passed, so a save refused for one of those runs no tool: its
    name and YAML, *locks*(connection, namespace, afresh) when given, its place (a new
    one's, with *adding*; else that it is there and, given *was*, unchanged) and its
    fields.
    The build holds no lock, so the write checks with same_grants() under its own."""
    if adding:
        check_name(name)
    with reading(connection):
        if merge:
            text = merged(namespace_yaml(connection, name) or "", text)
        namespace, _ = authored(name, text)
        afresh = granted_afresh(connection, namespace)
        if ignored := ignored_grants(connection, namespace):
            raise LibraryError("\n".join(ignored))
        if locks is not None:
            locks(connection, namespace, afresh)
        if adding:
            check_place(connection, name)
        else:
            check_unchanged(connection, name, was)
        check_fields(connection, namespace, adding)
    if afresh:
        build(afresh)
    return afresh


def ignored_grants(connection, namespace):
    """Why each tool *namespace* grants that its stored self doesn't, but runs would
    ignore, as it is left out (grants_ignored()), may not be granted: its why, and so."""
    left = read_tools(connection)[1]
    stored = read_namespaces(connection)[0].get(namespace.name, {}).get("tools", [])
    return [
        f"{left[tool]} Runs ignore a grant of it until it is fixed: fix or delete it"
        " in the Tools tab first."
        for tool in namespace.tools
        if tool in grants_ignored(left) and tool not in stored
    ]


def same_grants(built, locks, connection, namespace):
    """Raise unless *namespace* passes *locks* (as built_grants() takes them) and grants
    afresh the custom tools *built*, as the Library stands under the write lock: a
    write's allowed."""
    if locks is not None:
        locks(connection, namespace, built)
    if granted_afresh(connection, namespace) != built:
        raise LibraryError(CHANGED_MEANWHILE)


def store_tool(connection, name, block, source, allowed=None):
    """Store tool *name* as its config *block* and Python *source*, in place of one of its
    name, inside a transaction that keeps it only if the engine accepts the Library;
    *allowed*, given the connection under the write lock first, raises to refuse. The row
    it replaced as a change logs it (tool_text()), or for one the engine can't read a note
    saying why; None when it is new."""
    with writing(connection):
        if allowed is not None:
            allowed(connection)
        tools, left = read_tools(connection)
        before = left_out_note(left[name]) if name in left else None
        if name in tools:
            before = tool_text(name, dump(tools[name][0]), tools[name][1])
        connection.execute(UPSERT_TOOL, (name, dump(block), source))
        accepted(connection)
    return before


def delete_tool(connection, name):
    """Delete the Library's tool *name*, or raise LibraryError saying why not: a
    namespace grants it, or the Library holds no row for it. One left out, whose grants
    runs ignore, goes with its grants, which changes no run."""
    with writing(connection):
        namespaces = read_namespaces(connection)[0]
        granting = grants_of(namespaces, name)
        if name in grants_ignored(read_tools(connection)[1]):
            for each in granting:
                tools = [tool for tool in namespaces[each]["tools"] if tool != name]
                fields = dump(namespaces[each] | {"tools": tools})
                connection.execute(UPSERT_NAMESPACE, (each, fields))
            granting = []
        stored = connection.execute("SELECT 1 FROM tools WHERE name = ?", (name,))
        refusal = tool_refusal(name, granting, stored.fetchone() is not None)
        if refusal:
            raise LibraryError(refusal)
        connection.execute("DELETE FROM tools WHERE name = ?", (name,))


def tool_refusal(name, granting, stored):
    """Why tool *name* can't be deleted, as the namespaces *granting* grant it and the
    Library holds a row for it or not (*stored*); None when it can. A row under an app
    tool's name, as a shared file may hold, holds no grant: the app's tool keeps them."""
    if not stored and (name in BUILDERS or name in APP_TOOLS):
        whose = "the app's" if name in APP_TOOLS else "the engine's"
        return (
            f"{name} is {whose} own, not the Library's, so it can't be deleted; agents"
            " use it only where a namespace grants it."
        )
    if granting and name not in APP_TOOLS:
        return (
            f"{name} is granted in {', '.join(granting)}: take it out of their tools"
            " first."
        )
    return None if stored else f'The Library holds no tool "{name}" to delete.'


def delete_setting(connection, key):
    """Delete run setting *key*, or raise LibraryError saying why not: the engine reads it,
    as it does each it knows but one left out."""
    with writing(connection):
        left = read_run_settings(connection, read_namespaces(connection)[0])[1]
        if key in ENGINE_SETTINGS and key not in left:
            raise LibraryError(
                f"{key} is a run setting the engine reads, so it stays; only one it"
                " doesn't read can be deleted."
            )
        found = connection.execute("SELECT 1 FROM settings WHERE key = ?", (key,))
        if found.fetchone() is None:
            raise LibraryError(f'The Library holds no run setting "{key}".')
        connection.execute("DELETE FROM settings WHERE key = ?", (key,))


def delete_namespace(connection, name, decompositions):
    """Delete namespace *name* and its decompositions, as many as the caller was told it
    has, or raise LibraryError saying why not; if new chats started there, they start in root."""
    with writing(connection):
        if not has_namespace(connection, name):
            raise LibraryError(f'The Library has no namespace "{name}" to delete.')
        _, registry = config_and_registry(connection)
        spawns = own_spawns(registry, read_namespaces(connection)[1])
        refusal = namespace_refusal(spawns, name)
        if refusal:
            raise LibraryError(refusal)
        held = connection.execute(
            "SELECT count(*) FROM decompositions WHERE namespace = ?", (name,)
        ).fetchone()[0]
        if held != decompositions:
            raise LibraryError(
                f"{name} has {count(held, 'decomposition')} now, not {decompositions};"
                " look again before deleting it."
            )
        # Read first: once it is gone, an entry_namespace naming it reads as left out.
        entry = stored_settings(connection).get(ENTRY_NAMESPACE)
        connection.execute("DELETE FROM decompositions WHERE namespace = ?", (name,))
        connection.execute("DELETE FROM namespaces WHERE name = ?", (name,))
        if entry == name:
            connection.execute("DELETE FROM settings WHERE key = ?", (ENTRY_NAMESPACE,))
        accepted(connection)


def own_spawns(registry, left_out=()):
    """Each namespace's own spawn list, by name, each after the ones above it; each
    namespace *left_out*, as a shared file may hold, spawns nowhere, as no agent runs
    there."""
    spawns = {name: registry.config(name).spawn or [] for name in registry.names}
    spawns |= {name: [] for name in left_out if name not in spawns}
    return dict(sorted(spawns.items(), key=lambda each: chain_of(each[0])))


def namespace_refusal(spawns, name):
    """Why namespace *name* can't be deleted, as the namespaces stand, given as each one's
    own spawn list (*spawns*, as own_spawns() gives them); None when it can."""
    if name == ROOT:
        return "root can't be deleted: every namespace inherits from it."
    children = [child for child in spawns if child.startswith(f"{name}.")]
    if children:
        return f"Delete its namespaces first: {', '.join(children)}."
    # Else their spawn would name no namespace, which write_namespace refuses.
    spawners = [
        spawner
        for spawner, spawned in spawns.items()
        if spawner != name and name in spawned
    ]
    if spawners:
        return (
            f"First take {name} out of the spawn of {', '.join(spawners)}: they may"
            " spawn into it."
        )
    return None


def shown(value):
    if isinstance(value, dict | list):
        return flow(value)
    return "not set" if value is None else str(value)


def repl_text(repl):
    return flow(
        {"type": repl.type, **repl.model_dump(mode="json", exclude_defaults=True)}
    )


def count(number, noun):
    return f"{number} {noun}{'' if number == 1 else 's'}"


def cell(value):
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if isinstance(value, dict):
        return flow(value)[1:-1]
    return "" if value is None else str(value).strip()


def table(value):
    """A var that is records (a list of mappings, or mappings keyed by id) as columns and rows, else None."""
    if (
        isinstance(value, dict)
        and value
        and all(isinstance(row, dict) for row in value.values())
    ):
        value = [{"id": key, **row} for key, row in value.items()]
    if not (
        isinstance(value, list)
        and value
        and all(isinstance(row, dict) for row in value)
    ):
        return None
    columns = list(dict.fromkeys(key for row in value for key in row))
    return {
        "columns": columns,
        "rows": [[cell(row.get(column)) for column in columns] for row in value],
    }


def var_summary(value):
    if value is None:
        # What the agent's code gets: shown() would say "not set".
        return "None"
    rows = table(value)
    return count(len(rows["rows"]), "row") if rows else shown(value)


def located(value, level, namespace, verb="set"):
    """A namespace's value, and the level of its chain it comes from."""
    here = level == namespace
    source = f"{verb} here" if here else f"inherited from {level}"
    return {"value": value, "source": source, "here": here}


def backbone(resolved, run):
    if resolved.reasoner is not None:
        return located(
            shown(resolved.reasoner), resolved.sources.reasoner, resolved.name
        )
    if run.reasoner:
        return {"value": shown(run.reasoner), "source": "from the run settings"}
    return {"value": "chat", "source": "the run's default"}


def namespace_fields(registry, name, run, targets, lines):
    """Each field of namespace *name* as the engine resolves it, with where it comes from;
    *lines* as run_lines() gives them."""
    resolved = registry.resolve(name)
    sources = resolved.sources
    repl = repl_text(resolved.repl)
    if sources.repl == CONFIG_SOURCE:
        repl_item = {"value": repl, "source": "from the run settings"}
    else:
        repl_item = located(repl, sources.repl, name)
    if resolved.spawn is None:
        spawn = {"value": "any namespace", "source": "not set"}
    else:
        listed = [
            each if each in registry.names else f"{each} (not a namespace now)"
            for each in resolved.spawn
        ]
        spawn = located(", ".join(listed) or "nowhere: a leaf", sources.spawn, name)
    # Granted here when it lists the tool itself, though a level above grants it too.
    own_tools = registry.config(name).tools
    tools = [
        located(
            tool, name if tool in own_tools else sources.tools[tool], name, "granted"
        )
        for tool in resolved.tools
    ]
    variables = [
        located(f"{key} ({var_summary(value)})", sources.vars[key], name)
        for key, value in resolved.vars.items()
    ]
    suffixes = []
    for level in registry.chain(name):
        if level in sources.system_suffix:
            text = registry.config(level).system_suffix.strip()
            suffixes.append(located(text, level, name))
        if level in lines:
            item = located(lines[level], level, name)
            whose = "" if item["here"] else f", {item['source']}"
            suffixes.append(item | {"source": f"{ADDED_AT_RUN_TIME}{whose}"})
    fields = [
        {"label": "REPL", "items": [repl_item]},
        {"label": "Backbone", "items": [backbone(resolved, run)]},
        {"label": "May spawn into", "items": [spawn]},
        {"label": "Tools", "items": tools},
        {"label": "Variables", "items": variables},
        {"label": "System suffix", "items": suffixes},
    ]
    found = impossible_hand_offs(registry, name, targets)
    cant = Counter((to, why) for _, to, why in found)
    if cant:
        items = [
            cant_hand_off(registry, name, to, why, how_many)
            for (to, why), how_many in cant.items()
        ]
        fields.append({"label": "Hand-offs", "items": items})
    return fields


def cant_hand_off(registry, name, target, why, how_many):
    """What the panel says of the *how_many* worked examples namespace *name* is shown
    that hand off to *target*, where its agents can't."""
    instead = INSTEAD[hand_off_kind(registry, name, target)]
    value = (
        f"none to {target}: {why}. Its agents {instead} what"
        f" {count(how_many, 'worked example')} would hand there."
    )
    if target not in registry.names:
        return {"value": value, "source": "not in the Library"}
    return located(value, registry.resolve(name).sources.spawn, name)


def run_settings(read):
    """The run settings of the Library *read* (read_library()) as the panel lists them,
    each with where its value comes from; one left out reads as unset, says why, and may
    be deleted."""
    stored, left = read["settings"], read["settings_left"]
    defaults = V2Config().model_dump(mode="json")
    rows = []
    for key in dict.fromkeys([*SHOWN_SETTINGS, *stored, *left]):
        unread = key not in ENGINE_SETTINGS
        if key in left:
            source = f"the Library's is left out, so it reads as unset. {left[key]}"
        elif unread:
            source = UNREAD_SETTING
        else:
            source = "set in the Library" if key in stored else "the engine's default"
        value = shown(stored.get(key, defaults.get(key)))
        item = {"value": value, "source": source}
        rows.append({"label": key, "items": [item], "removable": unread or key in left})
    return rows


def source_description(source, factory, block):
    """What the agent is told a custom tool does, as far as its *source* says without
    running it: the description its *factory* gives Func(...), as text, a module's text
    constant or its *block*'s description setting, else, when it gives none, the
    docstring of the function or class it wraps. Empty when neither says, or the source
    doesn't parse, as a shared file may hold."""
    try:
        module = parsed_python(source)
    except LibraryError:
        return ""
    build = top_level(module.body).get(factory)
    if build is None:
        return ""
    texts = {
        target.id: node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    # A class factory's Func(...) is its own instance, which no call here shows.
    parameters = [] if isinstance(build, ast.ClassDef) else positional(build.args)
    settings = parameters[1].arg if len(parameters) > 1 else None
    for call in ast.walk(build):
        if isinstance(call, ast.Call) and called(call.func) == FUNC:
            # As Python finds a name in the factory: its own defs, then the module's.
            scope = top_level(module.body) | top_level(build.body)
            read = partial(read_setting, settings, block)
            return func_description(call, scope, texts, read)
    return ""


def positional(arguments):
    """The parameters a function's *arguments* take by position, in order."""
    return [*arguments.posonlyargs, *arguments.args]


def read_setting(settings, block, node):
    """*node* as a factory whose settings are its parameter *settings* reads it from
    *block*: for settings["description"], the setting; for settings.get("description",
    default), the setting or else the default; any other *node* as it is."""
    if isinstance(node, ast.Subscript):
        got, key, default = node.value, node.slice, node
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
    ):
        got, key = node.func.value, node.args[0]
        default = node.args[1] if len(node.args) > 1 else ast.Constant(None)
    else:
        return node
    named = isinstance(got, ast.Name) and got.id == settings
    if not (named and isinstance(key, ast.Constant) and key.value == DESCRIBED_BY):
        return node
    return ast.Constant(block[DESCRIBED_BY]) if DESCRIBED_BY in block else default


def top_level(body):
    """The functions and classes *body* defines itself, by name, the last of a name winning."""
    return {node.name: node for node in body if isinstance(node, DOCUMENTED)}


def func_description(call, scope, texts, read):
    """What the Func(value, description=None) *call* tells the agent: its description,
    as *read* (read_setting()) reads it, else value's docstring, where value names one of
    *scope*'s functions or classes."""
    given = argument(call, "description", 1)
    if given is not None:
        given = read(given)
    if given is not None and not is_none(given):
        text = literal_text(given, texts)
        return BUILT_DESCRIPTION.format(ast.unparse(given)) if text is None else text
    value = argument(call, "value", 0)
    wrapped = scope.get(called(value)) if value is not None else None
    return (wrapped and ast.get_docstring(wrapped)) or ""


def argument(call, name, position):
    """The node *call* passes as its parameter *name*, by keyword or at *position*."""
    given = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if given:
        return given[0]
    return call.args[position] if len(call.args) > position else None


def is_none(node):
    return isinstance(node, ast.Constant) and node.value is None


def called(node):
    """The name a call or a reference such as Func, primitives.Func or Counter() gives."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else None


def literal_text(node, texts):
    """*node*'s text when it is a string literal, or names one the module assigns in
    *texts*; None otherwise."""
    if isinstance(node, ast.Name):
        node = texts.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def origin(name, factory, block, source):
    """Where a tool comes from, as the panel shows it, and what the agent is told it does."""
    if name in APP_TOOLS:
        made = {"file": APP_TOOLS_MODULE, "defined_in": APP, "badge": "the app's"}
        return made, APP_TOOLS[name][1]
    if source is None:
        made = {
            "file": BUILDERS[factory].__module__,
            "defined_in": None,
            "badge": "built-in",
        }
        default = BUILTIN_DESCRIPTIONS[factory]
        if factory in EMPTY_TAKES_DEFAULT:
            return made, block.get(DESCRIBED_BY) or default
        return made, block.get(DESCRIBED_BY, default)
    made = {"file": block["factory_from"], "defined_in": "the Library", "badge": None}
    return made, source_description(source, factory, block) or NO_DESCRIPTION


def granted_in(registry, names, tool):
    """Those of the namespaces *names* that grant *tool* themselves."""
    return [each for each in names if tool in registry.config(each).tools]


def listed_tool(name, granting, stored, **shown):
    """Tool *name* as the panel lists it, granted in *granting*, and held by the Library
    (*stored*, so deletable) or by the engine or the app; *shown* gives what one left out,
    as a shared file may hold, lacks."""
    return {
        "name": name,
        "factory": None,
        "file": None,
        "defined_in": "the Library",
        "badge": None,
        "granted_in": granting,
        "description": None,
        "config": None,
        "stored": stored,
        "refusal": tool_refusal(name, granting, stored),
        "source": None,
        "problems": [],
    } | shown


def tool_rows(config, registry, names, sources):
    rows = []
    for name in declared_tool_names(config):
        block = config.tools.get(name, {})
        factory = block.get("factory", name)
        made, description = origin(name, factory, block, sources.get(name))
        row = listed_tool(
            name,
            granted_in(registry, names, name),
            name in sources,
            factory=factory,
            **made,
            description=description,
            config=dump({name: block}).rstrip() if block else None,
            source=sources.get(name),
        )
        rows.append(row)
    return rows


def variables(registry, names):
    """Each namespace's own vars that are tables, for the Data tab."""
    tables = [
        (key, name, table(value))
        for name in names
        for key, value in registry.config(name).vars.items()
    ]
    return [
        {"name": key, "where": name, "summary": shape(rows)} | rows
        for key, name, rows in tables
        if rows
    ]


def shape(rows):
    return (
        f"{count(len(rows['rows']), 'row')} · {count(len(rows['columns']), 'column')}"
    )


def parent_of(name):
    """The namespace *name* goes under, as the engine reads its name: root's is None."""
    if name == ROOT:
        return None
    head, dot, _ = name.rpartition(".")
    return head if dot else ROOT


def chain_of(name):
    """The namespaces *name* goes under, root first, then *name*."""
    chain = [name]
    while (parent := parent_of(chain[-1])) is not None:
        chain.append(parent)
    return chain[::-1]


def usable(name, left):
    """Whether a chat may run in namespace *name*, given those *left* out: the engine
    supplies a bare root when the file's is left out."""
    return name == ROOT or name not in left


def menu_of(connection):
    """What the namespace menu lists: each namespace, each after the ones above it, with
    whether a chat may run there, as the panel's rows say; and where new chats start."""
    read = read_library(connection)
    config, registry = run_config_and_registry(config_of(read))
    left = read["namespaces_left"]
    names = sorted({*registry.names, *left}, key=chain_of)
    return {
        "namespaces": [{"name": name, "usable": usable(name, left)} for name in names],
        "default_namespace": config.entry_namespace,
    }


def in_chain_order(registry):
    """The namespaces' names, each after the ones above it."""
    return sorted(registry.names, key=chain_of)


def namespace_rows(read, config, registry):
    """Each namespace of the Library *read* (read_library()) as the panel shows it: its
    depth, its fields as the engine resolves them, its own decompositions (each with its
    problems as a worked example), its own YAML (blank for one that sets nothing), how
    many worked examples elsewhere hand off to it, whether a chat may run there, and,
    for one the engine can't read, as a shared file may hold, why (problems). *config*
    and *registry* are as config_of(*read*) makes them."""
    decompositions = defaultdict(list)
    for row in read["decompositions"]:
        decompositions[row["namespace"]].append(row)
    left = read["namespaces_left"]
    targets = hand_off_targets(config)
    handed = Counter(hand_offs_made(registry, targets))
    readable = set(registry.names)
    spawns = own_spawns(registry, left)
    lines = run_lines(registry)
    return [
        {
            "name": name,
            "depth": len(chain_of(name)) - 1,
            "fields": (
                namespace_fields(registry, name, config, targets, lines)
                if name in readable
                else []
            ),
            "decompositions": checked(decompositions[name]),
            "yaml": own_yaml(read["namespace_yaml"].get(name, "")),
            "handed_to": handed[name],
            "usable": usable(name, left),
            "problems": left[name].splitlines() if name in left else [],
            "refusal": namespace_refusal(spawns, name),
        }
        for name in spawns
    ]


def hand_offs_made(registry, targets):
    """Where each stored worked example whose namespace can make its hand-off hands off,
    unless it is stored in the target or under it, which go with the target: what a
    Delete of the target warns of. *targets* as hand_off_targets() gives them."""
    for (namespace, _), target in targets.items():
        if target is None or namespace == target or namespace.startswith(f"{target}."):
            continue
        if unreachable(registry, namespace, target) is None:
            yield target


def checked(rows):
    """A namespace's stored decompositions, *rows* as read_decompositions() reads them,
    each with its problems as a worked example; or, for one left out, which agents aren't
    shown (read: false), why: its templates were checked as it was read."""
    return [
        {
            "name": row["name"],
            "yaml": row["yaml"],
            "problems": problems_of(row),
            "read": row["decomposition"] is not None,
        }
        for row in rows
    ]


def problems_of(row):
    decomposition = row["decomposition"]
    if decomposition is None:
        return row["why"].splitlines()
    return example_problems(decomposition)


def listed_tools(read, config, registry, names):
    """Each tool of the Library *read* (read_library()) as the panel lists it: those runs
    declare (tool_rows()), then each left out, saying why and where it is granted, so it
    can be taken out and the tool deleted. One runs declare all the same, the llm a model
    brings, is listed once, saying why its own row is left out."""
    sources = {name: source for name, (_, source) in read["tools"].items()}
    rows = {row["name"]: row for row in tool_rows(config, registry, names, sources)}
    for name, why in read["tools_left"].items():
        granting = grants_of(read["namespaces"], name)
        problems = why.splitlines()
        if name in rows:
            refusal = tool_refusal(name, granting, True)
            rows[name] |= {"stored": True, "refusal": refusal, "problems": problems}
            continue
        ignored = [GRANT_IGNORED] if granting else []
        problems = [*problems, *ignored]
        # Its grants go with it (delete_tool()).
        rows[name] = listed_tool(name, granting, True, problems=problems, refusal=None)
    return list(rows.values())


def panel(connection):
    """What the Library panel shows: run settings, namespaces as the engine resolves them, tools, tabular vars."""
    with reading(connection):
        read = read_library(connection)
        config, registry = run_config_and_registry(config_of(read))
        names = in_chain_order(registry)
        return {
            "default_namespace": config.entry_namespace,
            "run_settings": run_settings(read),
            "namespaces": namespace_rows(read, config, registry),
            "tools": listed_tools(read, config, registry, names),
            "variables": variables(registry, names),
        }
