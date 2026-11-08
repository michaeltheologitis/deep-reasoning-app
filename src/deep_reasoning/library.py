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
from functools import partial
from pathlib import Path

import deep_reasoner.llm
import jinja2
import structlog
import yaml
from deep_reasoner.config import load_cli_config
from deep_reasoner.namespaces import (
    CONFIG_SOURCE,
    ROOT,
    NamespaceConfig,
    NamespaceSpawnError,
    check_spawn,
)
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
from jinja2.sandbox import ImmutableSandboxedEnvironment
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
namespaces, library.tools() the tools. Write only what the user asked for, and answer by \
saying what changed, as the tool's result says. A change applies from the user's next \
message.
To change a worked example already stored, take its yaml from library.show(name) and \
edit it with .replace(...) in your cell; never type it out again, as its own repl tags \
would end your cell and run its code instead. To undo a change, find it with \
library.changes() and write its before back; never write a before from memory.
Only the user can change root, a namespace's repl or reasoner, which namespaces grant \
library, history or page, the system prompt or the run settings, or delete anything (in \
the Library panel: a namespace in its Namespaces tab, a decomposition in its \
Decompositions tab, a tool no namespace grants in its Tools tab): say so plainly, and \
how, as the tool's refusals and description do.
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
    ROOT: FRESH_ROOT,
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
# The librarian's: without the first, models write real <repl> tags into the example in
# their own cell, and the engine runs the example's code instead of the write.
LIBRARIANS_EXAMPLES = read_examples("worked_example_example.yaml", "undo_example.yaml")
# Root's are those two turns of each, under the same name: every namespace inherits them,
# but in historian its own take their place.
HAND_OFF_TURNS = 2
# Why a namespace's agents can't hand a question to *target*, as the panel and the worked
# example shown in place of the hand-off say.
CANT_REACH = jinja2.Template(
    "{% if gone %}there is no {{ target }} namespace now"
    "{% else %}this namespace's spawn list leaves it out{% endif %}",
    undefined=jinja2.StrictUndefined,
)
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
# Every run has one, so the Library is checked and shown as a run sees it: with the llm
# tool a model brings.
A_MODEL = {"model": "the chat's model"}
# Left to the Library, not a config: content (own tables), per-run paths, and the model and its key.
NOT_IMPORTED = {
    "namespaces",
    "namespaces_dir",
    "decompositions",
    "tools",
    "config_path",
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
"think", text}, {kind: "cell", code, output, status ("done", "failed" or "stopped"), \
agents: the nodes of the sub-agents it started}, {kind: "change", what, namespace, name, \
before (None when added), after} or {kind: "page", page, title} (a page opened for the \
user), each with its drive.
  history.errors(turn=None): what went wrong anywhere in message `turn`, or in every \
earlier message: each failed cell of any agent, the top one included, as turn, node, \
namespace (the agent's), task, index (its place in history.steps(turn, node)), error \
(what it raised) and code; then each agent that ended failed, stopped or exhausted, with \
index None. Start here for any question about an error.
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
research.papers): their tools, vars and system_suffix texts add up, and a decomposition \
of the same name lower down takes the place of its parent's there. Where a method takes \
yaml, give YAML text or a dict of plain values.
  library.show(name=None): every namespace, in brief: name, depth, sets (the fields it \
sets itself), tools (all its agents get, inherited ones included) and decompositions \
(each name, with how many problems it has as a worked example: 0 is ✓). Each text is \
cut after 200 characters, ending "…(+N chars)". show(name) gives namespace `name` whole: \
name, depth, yaml (the fields it sets itself, as change_namespace takes them), fields \
(each as the engine resolves it, with where it comes from) and decompositions (each with \
its name, yaml and problems).
  library.tools(name=None): every tool, in brief: name, kind ("custom", "built-in" or \
"the app's"), factory, settings, granted_in (the namespaces that grant it themselves), \
description and source (a custom tool's Python, else None), each text cut after 200 \
characters. tools(name) gives tool `name` whole.
  library.changes(turn=None): what this chat changed in the Library, in each of its \
messages (this one's so far included) or in message `turn` (numbered from 1, the chat's \
first): turn, message, what (namespace, decomposition or tool), namespace, name, before \
(None when it was added) and after, whole; a namespace's with changed (each field it \
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
so in your cell write the example's repl tags as [repl] and [/repl], in a raw string \
(r\"\"\"…\"\"\") so its code keeps the escapes you type; the tool writes the real tags. A \
yaml that already holds a real <repl> tag, as one from show(name) or changes() does, is \
written as it is, [repl] and all: to edit a stored decomposition, change its yaml in \
your cell (with .replace(...)) rather than typing it again. It is stored even when it \
isn't yet a worked example like that: the result's problems lists each way it isn't \
(empty when none), and until you write it again without them the Library panel marks it \
✗. The result's code_problems, when there are any, lists each turn whose code isn't \
valid Python, which agents copying it would copy too.
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
app's, a Python keyword or a name the REPL binds itself. The user changes namespaces in \
the Library panel's Namespaces tab (Edit YAML). There is no method for the system \
prompt or the run settings: they come from a config the user imports (deep-reasoning \
import), and only the user can change them.
Each write returns what it did, as {done, what, name} (and namespace, but for a tool); \
a refusal raises ValueError saying why. One that holds "…(+N chars)", where a brief \
list cut a text short, is refused: write the whole text."""
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
# A new namespace's name, from the page: plain words, joined by dots for a child.
NAMESPACE_NAME = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")
NAME_IT = 'Name the namespace, like "research"; "research.papers" goes under research.'
NAME_SHAPE = (
    "A namespace name is words of letters, digits, _ and -, joined by dots,"
    ' like "research.papers".'
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
# A var named as one of these hides it from the agent's code, as max: 10 breaks max(numbers).
BUILTIN_NAMES = frozenset(dir(builtins))
# Characters that don't show: controls and formats, such as a zero-width space.
INVISIBLE = ("Cc", "Cf")
# What the engine renders a turn's template against, beside prompt_template_variables.
FILLED_IN = ("task", "vars", "funcs", "coroutines")
# Never rendered, only parsed and compiled; with no globals, range() is an unknown name.
JINJA = Environment()
JINJA.globals.clear()
TEMPLATE = "the engine reads a turn holding {{ or {% as a Jinja template"
ESCAPE = (
    "Write {{ '{{' }} for a literal {{ and {{ '{%' }} for {%, or put the turn between"
    " {% raw %} and {% endraw %}."
)
NAMES_ONLY = (
    "a template may name only task, vars, funcs, coroutines and the run settings'"
    " prompt_template_variables, which the engine fills in"
)
# A shared file is a trust boundary: its templates must not reach Python internals.
# With no templates to include, an include fails as a render the engine words.
SANDBOXED = partial(ImmutableSandboxedEnvironment, loader=DictLoader({}))
# What a run renders no turn with: another template, as there are none, and a Python
# internal, which the sandbox refuses; the sandbox, not this, is the guard.
INCLUDES = {
    nodes.Include: "include",
    nodes.Extends: "extends",
    nodes.Import: "import",
    nodes.FromImport: "from",
}
REFUSED = (*INCLUDES, nodes.Getattr, nodes.Getitem, nodes.Filter)
# Compiling a template nested too deep raises Python's own errors, not Jinja's.
TEMPLATE_FAILURES = (TemplateError, SyntaxError, RecursionError, MemoryError)


class LibraryError(ValueError):
    """Something the Library refused, in words for the person who asked."""


class ChangedMeanwhile(LibraryError):
    """A save or delete refused because what it was opened on changed since."""


def changed_meanwhile(what):
    return ChangedMeanwhile(
        f"{what} was changed after you opened it (by the librarian, another tab or an"
        " import), so nothing was saved: saving would undo that change. Look at it as"
        " it is now, then save again."
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
    """Never run the file's own triggers and views: a shared file's could keep what is deleted."""
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, False)
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_VIEW, False)


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
    examples = [(ROOT, hand_off(text)) for text in HISTORIANS_EXAMPLES]
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


def stored_settings(connection):
    rows = connection.execute("SELECT key, yaml FROM settings ORDER BY rowid")
    return {key: yaml.safe_load(text) for key, text in rows if key not in CHATS_DECIDE}


def library_tools(connection):
    """The Library's own tools, each as its name, YAML and source. A row under an app
    tool's name, as a shared file may hold, is left out: the app's own takes its place."""
    rows = connection.execute("SELECT name, yaml, source FROM tools ORDER BY rowid")
    tools = []
    for name, text, source in rows:
        if name in APP_TOOLS:
            log.warning("library.tool_shadowed", tool=name)
            continue
        tools.append((name, text, source))
    return tools


def stored_config(connection):
    """The Library as a plain dict in the engine's config shape, the app's tools after its
    own; a custom tool's factory_from names its stored file."""
    tools = {name: yaml.safe_load(text) for name, text, _ in library_tools(connection)}
    tools |= {
        name: {"factory_from": str(APP_TOOLS_FILE), "factory": factory}
        for name, (factory, _) in APP_TOOLS.items()
    }
    decompositions = defaultdict(list)
    rows = connection.execute(
        "SELECT namespace, yaml FROM decompositions ORDER BY rowid"
    )
    for namespace, text in rows:
        decompositions[namespace].append(yaml.safe_load(text))
    rows = connection.execute("SELECT name, yaml FROM namespaces ORDER BY rowid")
    namespaces = {
        name: {**(yaml.safe_load(text) or {}), "decompositions": decompositions[name]}
        for name, text in rows
    }
    return {**stored_settings(connection), "tools": tools, "namespaces": namespaces}


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
    """Why *namespace*'s agents can't hand a question to namespace *target*, as the
    engine's check_spawn decides; None when they can."""
    if target not in registry.names:
        return CANT_REACH.render(gone=True, target=target)
    try:
        check_spawn(registry, namespace, target)
    except NamespaceSpawnError:
        return CANT_REACH.render(gone=False, target=target)
    return None


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
    return HISTORY_TOOL in registry.resolve(namespace).tools


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


def without_impossible_hand_offs(config):
    """*config* with each worked example that hands off where its namespace can't (one its
    spawn list leaves out, or one there is no longer) replaced, in that namespace, by what
    its agents do instead: copying the hand-off, the agent was refused, then handed the
    question to another namespace, which inherits the same example, until the depth ran
    out. A namespace holds an example only where what it inherits differs from what it
    should see: the replacement, or the original back below one that can't."""
    try:
        run, registry = run_config_and_registry(config)
        targets = hand_off_targets(run)
        swaps = {
            name: {
                example.name: saying_so(registry, name, example, target, why)
                for example, target, why in impossible_hand_offs(
                    registry, name, targets
                )
            }
            for name in registry.names
        }
    except ValueError as error:
        # The worker refuses such a Library itself, in its own words.
        log.warning("library.hand_offs_not_checked", error=str(error))
        return config
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
    the variables of the worker's environment that hold keys.
    """
    tools_dir = Path(tools_dir).resolve()
    tools_dir.mkdir(parents=True, exist_ok=True)
    with closing(open_library(library_path)) as connection:
        config = stored_config(connection)
        tools = library_tools(connection)
    grants = granted(config["namespaces"])
    config["tools"] = {
        name: block for name, block in config["tools"].items() if name in grants
    }
    for name, _, source in tools:
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
    return without_impossible_hand_offs(config) | {"entry_namespace": namespace}


def default_namespace(library_path):
    """Where a new chat starts: the run settings' entry_namespace, root unless one is set."""
    with closing(open_library(library_path)) as connection:
        return stored_settings(connection).get(ENTRY_NAMESPACE, ROOT)


def has_namespace(connection, name):
    found = connection.execute("SELECT 1 FROM namespaces WHERE name = ?", (name,))
    return found.fetchone() is not None


def namespace_yaml(connection, name):
    """Namespace *name*'s own YAML as stored; None when the Library has no namespace
    *name*."""
    found = connection.execute(
        "SELECT yaml FROM namespaces WHERE name = ?", (name,)
    ).fetchone()
    return None if found is None else found[0]


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


def tool_row(name, block, config_dir):
    """A tool's row: its config block, and for a tool of your own its Python source."""
    if name in APP_TOOLS:
        raise LibraryError(
            f'Tool "{name}" has the name of one of the app\'s own tools; rename yours.'
        )
    factory = block.get("factory", name)
    factory_from = block.get("factory_from")
    if not isinstance(factory, str) or not isinstance(factory_from, str | None):
        raise LibraryError(f'Tool "{name}": factory and factory_from should be text.')
    if factory_from is None:
        if factory not in BUILDERS:
            raise LibraryError(
                f'Tool "{name}" names factory "{factory}", which is not built in; '
                "give its block a factory_from: file."
            )
        if factory == "llm":
            # It runs on the chat's model, as the chat does; the config's model or preset may not exist there.
            block = {key: value for key, value in block.items() if key != "model"}
        return name, dump(block), None
    stored_file = tool_file(name)
    file = config_dir / factory_from
    source = file.read_text(encoding="utf-8")
    parsed_python(source, str(file))
    return name, dump({**block, "factory_from": stored_file}), source


def store_decompositions(connection, namespace, decompositions):
    rows = [(namespace, each.name, dump(each.model_dump())) for each in decompositions]
    connection.executemany(UPSERT_DECOMPOSITION, rows)


def import_config(connection, path):
    """Read the deep_reasoner config at *path* into the Library, replacing same-named items.

    It takes the config's namespaces (inline and from its namespaces_dir) with their
    decompositions, its tools with the Python files their factory_from: names, and its run
    settings but for the model and how to reach it. Top-level decompositions are left out:
    with namespaces they are only `dr --main-decomposition` openings, which a chat has none of.
    All of it lands, or none.
    """
    config = load_cli_config(path, schema=V2Config)
    registry = build_namespace_registry(config)
    settings = config.model_dump(exclude_unset=True, exclude=NOT_IMPORTED)
    tools = [
        tool_row(name, block, config.config_dir) for name, block in config.tools.items()
    ]
    # A root that sets nothing, like the bare one the engine adds, must not blank the Library's.
    namespaces = [
        namespace
        for namespace in map(registry.config, registry.names)
        if namespace.name != ROOT
        or namespace.decompositions
        or namespace.model_fields_set - {"name", "decompositions"}
    ]
    with writing(connection):
        connection.executemany(
            UPSERT_SETTING, [(k, dump(v)) for k, v in settings.items()]
        )
        connection.executemany(UPSERT_TOOL, tools)
        for namespace in namespaces:
            own = namespace.model_dump(
                exclude_unset=True, exclude={"name", "decompositions"}
            )
            connection.execute(UPSERT_NAMESPACE, (namespace.name, dump(own)))
            store_decompositions(connection, namespace.name, namespace.decompositions)
        _, registry = check(connection)
        check_backbones(registry, registry.names)
    return {
        "namespaces": len(namespaces),
        "decompositions": sum(len(each.decompositions) for each in namespaces),
        "tools": len(tools),
    }


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
    """The YAML *text* as data, or a LibraryError saying where it doesn't parse."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f"line {mark.line + 1}: " if mark else ""
        reason = getattr(error, "problem", None) or error
        raise LibraryError(
            f"The YAML doesn't parse ({where}{reason}). Check the indentation and quoting on that line."
        ) from error


def parsed(text):
    """The decomposition the YAML *text* describes, checked by the engine's own model."""
    data = loaded(text)
    if not isinstance(data, dict):
        raise LibraryError(
            "The YAML should be a mapping: a name: line, then messages:, as in the template."
        )
    try:
        decomposition = Decomposition.model_validate(data)
    except ValidationError as error:
        raise LibraryError("\n".join(map(problem, error.errors()))) from error
    name = decomposition.name
    if not name.strip():
        raise LibraryError(BLANK_NAME)
    if name != name.strip():
        raise LibraryError(
            f'The name "{name}" starts or ends with a space; take it out, so it reads as it'
            " looks."
        )
    hidden = [char for char in name if unicodedata.category(char) in INVISIBLE]
    if hidden:
        raise LibraryError(
            f"The name holds a character you can't see (U+{ord(hidden[0]):04X}); take it"
            " out, so it reads as it looks."
        )
    return decomposition


def checked_text(text):
    """Raise unless *text* is short enough to be a decomposition's YAML."""
    if len(text) > MAX_YAML_CHARS:
        raise LibraryError(
            f"A decomposition's YAML may be at most {MAX_YAML_CHARS:,} characters; this"
            f" has {len(text):,}."
        )


def sandbox_templates():
    """Make every Jinja render the engine does in this process sandboxed: each process
    that runs the engine calls it first."""
    deep_reasoner.llm._JinjaEnv = SANDBOXED


def template_names(config):
    """The names a turn's template may read: what the engine fills in, and the run
    settings' prompt_template_variables in *config*."""
    return {*FILLED_IN, *config.prompt_template_variables}


def first_use(template, name):
    """The line of *template* (parsed) where *name* is first read."""
    uses = [
        each.lineno
        for each in template.find_all(nodes.Name)
        if each.name == name and each.ctx == "load"
    ]
    return min(uses)


def internal_read(node):
    """The Python internal *node* reads, a name starting with _, which a run's sandbox
    refuses; None for none."""
    if isinstance(node, nodes.Getattr):
        name = node.attr
    elif isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        name = node.arg.value
    else:
        return None
    return name if isinstance(name, str) and name.startswith("_") else None


def refused(node):
    """Why a worked example may not hold *node*, in words; None when it may."""
    if type(node) in INCLUDES:
        tag = INCLUDES[type(node)]
        return (
            f"includes another template ({{% {tag} %}}), which a worked example can't"
        )
    if isinstance(node, nodes.Filter) and node.name == "attr":
        return "reads an attribute through |attr, which a worked example can't; write .name instead"
    if internal := internal_read(node):
        return f"reaches into Python internals ({internal}), which a run refuses"
    return None


def first_refused(template):
    """The line of *template* (parsed) where it first holds what a worked example may
    not, and why; None when it holds none. Of a chain such as a._b._c, the first read."""
    for node in template.find_all(REFUSED):
        if refused(node):
            while (inner := getattr(node, "node", None)) is not None and refused(inner):
                node = inner
            return node.lineno, refused(node)
    return None


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
    problems = template_problems(decomposition, template_names(config))
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


def stored_problems(text, names):
    """The problems of the decomposition stored as YAML *text*: its shape's, as a shared
    file may hold anything, or else its turns', as a worked example and as templates
    reading only *names*."""
    try:
        decomposition = parsed(text)
    except LibraryError as error:
        return str(error).splitlines()
    return example_problems(decomposition) + template_problems(decomposition, names)


def add_decomposition(connection, namespace, text, replace=False, allowed=None):
    """Store the decomposition written as YAML *text* under *namespace*, or raise
    LibraryError saying why not: one of its name there is refused, or with *replace* put in
    its place. *allowed*, given the connection under the write lock first, raises to refuse.
    A worked example with problems is stored all the same, so a draft can be saved.
    The stored row with its problems, and the YAML it replaced (None when it is new)."""
    checked_text(text)
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


def stored_decomposition(connection, namespace, name):
    """The YAML of *namespace*'s decomposition *name*, None when it has none."""
    found = connection.execute(
        "SELECT yaml FROM decompositions WHERE namespace = ? AND name = ?",
        (namespace, name),
    ).fetchone()
    return None if found is None else found[0]


def already_has(namespace, name):
    return LibraryError(
        f'{namespace} already has "{name}"; rename it, pick another namespace, or Edit'
        " that one."
    )


def edit_decomposition(connection, namespace, name, was, text):
    """Put the decomposition written as YAML *text* in the place of *namespace*'s
    decomposition *name*, which the person opened as *was*; it may take another name
    that *namespace* doesn't hold. Raises LibraryError saying why not, and
    ChangedMeanwhile if *name* is no longer stored as *was*."""
    checked_text(text)
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


def delete_decomposition(connection, namespace, name, was):
    """Delete *namespace*'s decomposition *name*, which the person was shown as *was*;
    ChangedMeanwhile if it is no longer stored so."""
    with writing(connection):
        deleted = connection.execute(
            "DELETE FROM decompositions WHERE namespace = ? AND name = ? AND yaml = ?",
            (namespace, name, was),
        )
        if deleted.rowcount == 0:
            raise changed_meanwhile(f'"{name}" in {namespace}')


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
    data = loaded(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise LibraryError(
            'The YAML should be a mapping of fields, like "tools: [llm]".'
        )
    problems = [NAMED_ELSEWHERE[key] for key in NAMED_ELSEWHERE if key in data]
    fields = {key: value for key, value in data.items() if key not in NAMED_ELSEWHERE}
    try:
        namespace = NamespaceConfig.model_validate(fields | {"name": name})
    except ValidationError as error:
        problems += map(namespace_problem, error.errors())
        raise LibraryError("\n".join(problems)) from error
    if problems:
        raise LibraryError("\n".join(problems))
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
    if name in BUILTIN_NAMES:
        return (
            f"Var \"{name}\" would hide Python's own {name}, which the agent's code may"
            f' use; rename it, like "{name}_value".'
        )
    return None


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


def check_fields(connection, namespace):
    """Raise unless *namespace*'s vars are well named and every namespace it may spawn
    into is in the Library, or is itself."""
    problems = list(filter(None, map(var_problem, namespace.vars)))
    if problems:
        raise LibraryError("\n".join(problems))
    # The engine lets a spawn name no namespace, and fails only when the agent spawns there.
    missing = [
        each
        for each in namespace.spawn or []
        if each != namespace.name and not has_namespace(connection, each)
    ]
    if missing:
        raise LibraryError(
            "\n".join(
                f'spawn names "{each}", which is not a namespace in the Library.'
                for each in missing
            )
        )


def write_namespace(connection, namespace, text):
    """Store *namespace* as its YAML *text*, inside a transaction that keeps it only if
    its fields pass check_fields, the engine accepts the Library, and there and below
    each backbone is one the engine builds and no tool takes a var's place. A namespace
    elsewhere is not its to answer for."""
    check_fields(connection, namespace)
    connection.execute(UPSERT_NAMESPACE, (namespace.name, text))
    _, registry = accepted(connection)
    below = [each for each in registry.names if namespace.name in registry.chain(each)]
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
        write_namespace(connection, namespace, stored)
    return stored


def check_name(name):
    """Raise unless *name* names a namespace: plain words, joined by dots for a child."""
    if not name:
        raise LibraryError(NAME_IT)
    if not NAMESPACE_NAME.fullmatch(name):
        raise LibraryError(NAME_SHAPE)


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
    except (LibraryError, RecursionError):
        return {}
    return fields if isinstance(fields, dict) else {}


def merged(before, text):
    """A namespace's own YAML *before* with each field the YAML *text* sets in place of
    its own, and each it sets to null taken out; the rest kept, as typed when none
    changes."""
    given = loaded(text)
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise LibraryError(
            'The YAML should be a mapping of fields, like "tools: [llm]".'
        )
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
    if len(together) > MAX_YAML_CHARS:
        raise LibraryError(
            f"A namespace's YAML may be at most {MAX_YAML_CHARS:,} characters; with"
            f" this change it would be {len(together):,}."
        )
    return together


def change_namespace(connection, name, text, allowed=None, was=None, merge=False):
    """Give namespace *name* the fields its YAML *text* gives in place of its own, keeping
    its decompositions, or raise LibraryError saying why not; with *merge*, only those
    fields, keeping the rest. *allowed* as add_namespace takes it. Given *was*, the YAML
    the person opened it on, ChangedMeanwhile unless it is still stored so. The YAML it
    replaced, and the YAML stored."""
    with writing(connection):
        if merge:
            text = merged(namespace_yaml(connection, name) or "", text)
        namespace, stored = authored(name, text)
        if allowed is not None:
            allowed(connection, namespace)
        before = check_namespace(connection, name)
        if was is not None and own_yaml(before) != was:
            raise changed_meanwhile(f"Namespace {name}")
        write_namespace(connection, namespace, stored)
    return before, stored


def store_tool(connection, name, block, source, allowed=None):
    """Store tool *name* as its config *block* and Python *source*, in place of one of its
    name, inside a transaction that keeps it only if the engine accepts the Library;
    *allowed*, given the connection under the write lock first, raises to refuse. The row
    it replaced, as its block's YAML and its source; None when it is new."""
    with writing(connection):
        if allowed is not None:
            allowed(connection)
        found = connection.execute(
            "SELECT yaml, source FROM tools WHERE name = ?", (name,)
        ).fetchone()
        connection.execute(UPSERT_TOOL, (name, dump(block), source))
        accepted(connection)
    return found


def delete_tool(connection, name):
    """Delete the Library's tool *name*, or raise LibraryError saying why not: a
    namespace grants it."""
    with writing(connection):
        _, registry = config_and_registry(connection)
        granting = granted_in(registry, in_chain_order(registry), name)
        if granting:
            raise LibraryError(
                f"{name} is granted in {', '.join(granting)}: take it out of their tools"
                " first."
            )
        deleted = connection.execute("DELETE FROM tools WHERE name = ?", (name,))
        if deleted.rowcount == 0:
            raise LibraryError(f'The Library holds no tool "{name}" to delete.')


def delete_setting(connection, key):
    """Delete run setting *key*, or raise LibraryError saying why not: the engine reads it."""
    if key in ENGINE_SETTINGS:
        raise LibraryError(
            f"{key} is a run setting the engine reads, so it stays; only one it doesn't"
            " read can be deleted."
        )
    with writing(connection):
        found = connection.execute("SELECT 1 FROM settings WHERE key = ?", (key,))
        if found.fetchone() is None:
            raise LibraryError(f'The Library holds no run setting "{key}".')
        connection.execute("DELETE FROM settings WHERE key = ?", (key,))


def delete_namespace(connection, name, decompositions):
    """Delete namespace *name* and its decompositions, as many as the caller was told it
    has, or raise LibraryError saying why not; if new chats started there, they start in root."""
    if name == ROOT:
        raise LibraryError("root can't be deleted: every namespace inherits from it.")
    with writing(connection):
        check_namespace(connection, name)
        rows = connection.execute("SELECT name, yaml FROM namespaces")
        namespaces = {each: yaml.safe_load(text) or {} for each, text in rows}
        children = [each for each in namespaces if each.startswith(f"{name}.")]
        if children:
            raise LibraryError(f"Delete its namespaces first: {', '.join(children)}.")
        # Else their spawn would name no namespace, which write_namespace refuses.
        spawners = [
            each
            for each, fields in namespaces.items()
            if each != name and name in (fields.get("spawn") or [])
        ]
        if spawners:
            raise LibraryError(
                f"First take {name} out of the spawn of {', '.join(spawners)}:"
                " they may spawn into it."
            )
        held = connection.execute(
            "SELECT count(*) FROM decompositions WHERE namespace = ?", (name,)
        ).fetchone()[0]
        if held != decompositions:
            raise LibraryError(
                f"{name} has {count(held, 'decomposition')} now, not {decompositions};"
                " look again before deleting it."
            )
        connection.execute("DELETE FROM decompositions WHERE namespace = ?", (name,))
        connection.execute("DELETE FROM namespaces WHERE name = ?", (name,))
        if stored_settings(connection).get(ENTRY_NAMESPACE) == name:
            connection.execute("DELETE FROM settings WHERE key = ?", (ENTRY_NAMESPACE,))
        accepted(connection)


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


def namespace_fields(registry, name, run, targets):
    """Each field of namespace *name* as the engine resolves it, with where it comes from."""
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
        spawn = located(
            ", ".join(resolved.spawn) or "nowhere: a leaf", sources.spawn, name
        )
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
    suffixes = [
        located(registry.config(level).system_suffix.strip(), level, name)
        for level in sources.system_suffix
    ]
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


def run_settings(connection):
    stored = stored_settings(connection)
    defaults = V2Config().model_dump(mode="json")
    rows = []
    for key in dict.fromkeys([*SHOWN_SETTINGS, *stored]):
        unread = key not in ENGINE_SETTINGS
        source = "set in the Library" if key in stored else "the engine's default"
        value = shown(stored.get(key, defaults.get(key)))
        item = {"value": value, "source": UNREAD_SETTING if unread else source}
        rows.append({"label": key, "items": [item], "removable": unread})
    return rows


def class_docstring(source):
    """The first class's docstring: what the agent is told when a factory returns one and gives no description.
    Empty for source that doesn't parse, as a shared file may hold, so the panel still lists the tool."""
    try:
        body = parsed_python(source).body
    except LibraryError:
        return ""
    classes = [node for node in body if isinstance(node, ast.ClassDef)]
    return (ast.get_docstring(classes[0]) if classes else None) or ""


def origin(name, factory, block, source):
    """Where a tool comes from, as the panel shows it, and what the agent is told it does
    unless its block says."""
    if name in APP_TOOLS:
        made = {"file": APP_TOOLS_MODULE, "defined_in": APP, "badge": "the app's"}
        return made, APP_TOOLS[name][1]
    if source is None:
        made = {
            "file": BUILDERS[factory].__module__,
            "defined_in": None,
            "badge": "built-in",
        }
        return made, BUILTIN_DESCRIPTIONS[factory]
    made = {"file": block["factory_from"], "defined_in": "the Library", "badge": None}
    return made, class_docstring(source)


def granted_in(registry, names, tool):
    """Those of the namespaces *names* that grant *tool* themselves."""
    return [each for each in names if tool in registry.config(each).tools]


def tool_rows(config, registry, names, sources):
    rows = []
    for name in declared_tool_names(config):
        block = config.tools.get(name, {})
        factory = block.get("factory", name)
        made, description = origin(name, factory, block, sources.get(name))
        row = {
            "name": name,
            "factory": factory,
            **made,
            "granted_in": granted_in(registry, names, name),
            "description": block.get("description") or description,
            "config": dump({name: block}).rstrip() if block else None,
            # Deletable: the Library holds it, not the engine or the app.
            "stored": name in sources,
            "source": sources.get(name),
        }
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


def in_chain_order(registry):
    """The namespaces' names, each after the ones above it."""
    return sorted(registry.names, key=registry.chain)


def namespace_rows(connection, config, registry):
    """Each namespace as the panel shows it: its depth, its fields as the engine resolves
    them, its own decompositions (each with its problems as a worked example), its own
    YAML, and how many worked examples elsewhere hand off to it. *config* and *registry*
    are read in the same read (reading()) as its rows."""
    decompositions = defaultdict(list)
    rows = connection.execute(
        "SELECT namespace, name, yaml FROM decompositions ORDER BY rowid"
    )
    for namespace, name, text in rows:
        decompositions[namespace].append({"name": name, "yaml": text})
    # What each one's Edit YAML opens on: blank for a namespace that sets nothing.
    rows = connection.execute("SELECT name, yaml FROM namespaces")
    own = {name: own_yaml(text) for name, text in rows}
    names = template_names(config)
    targets = hand_off_targets(config)
    handed = Counter(hand_offs_made(registry, targets))
    return [
        {
            "name": name,
            "depth": len(registry.chain(name)) - 1,
            "fields": namespace_fields(registry, name, config, targets),
            "decompositions": checked(
                decompositions[name], config.namespaces.get(name), names
            ),
            "yaml": own.get(name, ""),
            "handed_to": handed[name],
        }
        for name in in_chain_order(registry)
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


def checked(rows, namespace, names):
    """A namespace's stored decompositions, *rows*, each with its problems as a worked
    example and as templates reading only *names*, as a shared file may hold one the
    engine can't render, from *namespace*, the engine's reading of the same rows. Under a
    root the file lacks, which the engine reads none of, they are parsed here."""
    if namespace is None:
        return [row | {"problems": stored_problems(row["yaml"], names)} for row in rows]
    pairs = zip(rows, namespace.decompositions, strict=True)
    return [
        row | {"problems": example_problems(each) + template_problems(each, names)}
        for row, each in pairs
    ]


def panel(connection):
    """What the Library panel shows: run settings, namespaces as the engine resolves them, tools, tabular vars."""
    with reading(connection):
        config, registry = config_and_registry(connection)
        names = in_chain_order(registry)
        sources = {name: source for name, _, source in library_tools(connection)}
        return {
            "default_namespace": config.entry_namespace,
            "run_settings": run_settings(connection),
            "namespaces": namespace_rows(connection, config, registry),
            "tools": tool_rows(config, registry, names, sources),
            "variables": variables(registry, names),
        }
