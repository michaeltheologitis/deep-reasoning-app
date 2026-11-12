"""One run of deep_reasoner, as `python -P -m deep_reasoning.worker <run folder>`, in the
chat's folder.

It runs the run folder's spec.yaml and writes the run's events to stdout, one JSON line each.
It takes one command on stdin, `stop`. Anything else the run prints lands on stderr,
which the server sends to the folder's worker.log, and so do the app's own log records,
which the engine would copy into the output of the cell that logged them.
"""

import ast
import asyncio
import inspect
import json
import logging
import math
import os
import re
import select
import signal
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NoReturn, TextIO

import structlog
import yaml
from deep_reasoner.core import configure_structlog_fixture
from deep_reasoner.namespaces import NamespaceRegistry
from deep_reasoner.tools import ToolBuildError
from deep_reasoner.v2 import agent
from deep_reasoner.v2.agent import _parts
from deep_reasoner.v2.cli import (
    V2Config,
    build_namespace_registry,
    build_reasoner,
    close_run,
)
from deep_reasoner.v2.decompositions import as_cli_act
from deep_reasoner.v2.messages import Decomposition, NoCodeBlock, code, func
from genai_prices import Usage, calc_price
from openai import APIStatusError
from openai.resources.chat.completions import AsyncCompletions

from deep_reasoning.app_tools import CLAUDE_CODE
from deep_reasoning.library import (
    EXAMPLES_STORED_IN,
    TOLD_LAST,
    count,
    granted_in,
    in_chain_order,
    plural,
    sandbox_templates,
)
from deep_reasoning.runs import (
    CHANGE_FIELDS,
    FINAL_ANSWER,
    LEFT_OUT,
    PAGE_FIELDS,
    SPEC,
    escaped,
    raised_error,
)

log = structlog.get_logger()

# The logger of every module of the app's, the library tool's included.
APP_LOGGER = __package__
CLOSE_RUN_WATCHDOG_S = 2.0
STDIN_READ_SIZE = 4096
# Held while the server's commands are read: by the stdin thread, or as the root starts.
TAKING_COMMANDS = threading.Lock()
REPL_BLOCK = re.compile(r"<repl>.*?(?:</repl>|$)", re.DOTALL)
THINK_TAG = re.compile(r"</?think>")
OBSERVATION = re.compile(r"\A<observation>\n?(.*?)\n?</observation>\Z", re.DOTALL)
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
# The worked examples left out of the agent starting in this context, as it renders
# them, each as its LEFT_OUT event says it.
LEFT_OUT_NOW: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "left_out_now", default=()
)
WHY_CHARS = 300
# The engine's message when a reply has no <repl> block, word for word, and the same
# sent as the app's: read as the user's, it had the model apologise to the user for it.
NO_REPL_NOTE = (
    "Your response did not contain a `<repl>...</repl>` block. Please include one."
)
NO_REPL_TURN = {"role": "user", "content": NO_REPL_NOTE}
# Answered as a note, the reply became the user's answer ("Understood."), so it says
# what to do with the reply the user never saw.
APPS_NO_REPL_TURN = {
    "role": "user",
    "content": "(An automated note from the app, not the user, who sees none of this: "
    f"{NO_REPL_NOTE}) The user saw none of what you just wrote: if it was your answer,"
    " put that answer, as you wrote it, in FinalAnswer(...) in a <repl> block, and never"
    " reply to this note itself.",
}
# A provider's failure or refusal that ends a run is led by what it means, on a line of
# its own beginning PROVIDER_LEAD, which the page reads as the app's words (failure_detail()).
PROVIDER_LEAD = "The model provider"
SERVER_ERROR = 500
BAD_REQUEST = 400
# The code of a 400 for a prompt the provider refuses for what it holds, as OpenAI's.
REFUSED_CODES = frozenset({"invalid_prompt"})
KEPT = "The {changes} this run made to the Library {are} kept."
AGAIN = "Try again, or ask in smaller steps."
# The worker's own line leading a run out of turns, as the page reads it (tree.js).
OUT_OF_TURNS = "The agent used all its turns without answering."
SMALLER_STEPS = "Ask again in smaller steps."
NOTHING_PRINTED = frozenset({"", "(no output)"})
# Shorter outputs ("True", "nothing changed: …") repeat in ordinary work.
REREAD_CHARS = 200
MOSTLY_PRINTED_BEFORE = 0.8
# Read as the user's message, a printed list got the answer "Understood.".
OWN_OUTPUT_NOTE = (
    "(An automated note from the app, not the user: this is what your last cell printed,"
    " which the user never sees; you keep it and its variables.)"
)
# Helpers that printed a file started the task again from the top, up to 30 times.
REPEATED_NOTE = (
    "(An automated note from the app, not the user: this cell printed what an earlier"
    " one already had, and you still have that and its variables: don't read or work it"
    " out again. Answer from it now with FinalAnswer, or go on to the next step from it.)"
)
# A few hundred rows go whole; a till export printed whole overran the model's context.
SENT_OUTPUT_CHARS = 50_000
# Of each end kept: whole lines must keep more than this, or the cut goes mid-line.
WHOLE_LINES_KEEP = 0.5
CUT_NOTE = (
    "(An automated note from the app, not the user: the middle {left:,} of this output's"
    " {total:,} characters are left out here{how}, too many to send you. Print only what"
    " you need from it: a summary, a count, or the rows you want.)"
)
MID_LINE = ", cut mid-line"
TURNS_LEFT_WARNED = 3
TURNS_LEFT_NOTE = (
    f"(An automated note from the app, not the user: you have {TURNS_LEFT_WARNED} turns"
    " left. Finish now with FinalAnswer, saying what you did, what you found and what is"
    " left to do.)"
)


def source_of(reply: str) -> str | None:
    """The <repl> block deep_reasoner would run from this reply, if it has one."""
    try:
        return code(reply).source
    except NoCodeBlock:
        return None


def thought_of(reply: str) -> str:
    """The reply without its <repl> block and <think> tags."""
    return THINK_TAG.sub("", REPL_BLOCK.sub("", reply)).strip()


def output_of(observation: str) -> str:
    """What a cell printed, without the <observation> tags around it."""
    match = OBSERVATION.match(observation)
    return match.group(1) if match else observation


def final_answer_of(output: str) -> str | None:
    """The repr after the last line that starts with "FinalAnswer: ", to the end."""
    at = ("\n" + output).rfind("\n" + FINAL_ANSWER)
    if at < 0:
        return None
    return output[at + len(FINAL_ANSWER) :].rstrip("\n")


def unquote(answer_repr: str) -> str:
    """A FinalAnswer repr, unquoted when it is a Python string literal."""
    try:
        value = ast.literal_eval(answer_repr)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return answer_repr
    return value if isinstance(value, str) else answer_repr


def cost_of(model: str, usage: dict[str, Any], base_url: str) -> float | None:
    """One model call's cost in USD: the provider's own figure when it sends one, else
    genai-prices' price of the model at the vendor whose address *base_url* is, with cached
    input at its rate; None for any other address, as a server of your own may serve a
    model under a vendor's name, or a model the vendor doesn't price, or a figure below
    zero or past any number, which prices nothing."""
    if usage.get("cost") is not None:
        cost = float(usage["cost"])
        return cost if 0 <= cost < math.inf else None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    tokens = Usage(
        input_tokens=usage.get("prompt_tokens") or 0,
        cache_read_tokens=cached,
        output_tokens=usage.get("completion_tokens") or 0,
    )
    try:
        return float(calc_price(tokens, model, provider_api_url=base_url).total_price)
    except LookupError:
        return None


def for_claude(message: dict[str, str]) -> dict[str, str]:
    """An earlier message as a Claude session is shown it: an assistant turn's <repl> cell
    written as the command the session runs instead."""
    if message["role"] != "assistant":
        return message
    return message | {"content": as_cli_act(message["content"])}


def detail_of(error: BaseException) -> str:
    """An exception as deep_reasoner words agent.end's detail."""
    return f"{type(error).__name__}: {error}"


def provider_lead(error: BaseException) -> str | None:
    """What a provider's *error* means, in a sentence: its failure (5xx), after how long
    the call waited, its retries included, or its refusal of the prompt (a 400 of
    REFUSED_CODES); None for any other error."""
    status = error.status_code if isinstance(error, APIStatusError) else None
    if status is not None and status >= SERVER_ERROR:
        waited = getattr(error, "seconds_waited", None)
        after = "" if waited is None else f" after {math.ceil(waited)} s of waiting"
        return f"{PROVIDER_LEAD} failed ({status}){after}."
    if status == BAD_REQUEST and getattr(error, "code", None) in REFUSED_CODES:
        return f"{PROVIDER_LEAD} refused this prompt."
    return None


def failure_detail(error: BaseException, changes: int) -> str:
    """Why a run ended by *error*: a provider's, led by what it means (provider_lead()),
    then the *changes* to the Library the run made, which stay, and what to do, then the
    provider's own words; any other error in the engine's words (detail_of())."""
    words = detail_of(error)
    lead = provider_lead(error)
    return words if lead is None else led(lead, changes, AGAIN, words)


def provider_led(detail: str | None) -> str | None:
    """An agent's end *detail*, on a line after what the provider's error that ended it
    means, as a run's: the engine logs a failed drive's end while handling its error."""
    error = sys.exc_info()[1]
    lead = provider_lead(error) if error is not None else None
    if lead is None or detail != detail_of(error):
        return detail
    return f"{lead}\n{detail}"


def exhausted_detail(changes: int, sentence: str) -> str:
    """Why a run ended without an answer: its agent used all its turns; then the
    engine's *sentence*."""
    return led(OUT_OF_TURNS, changes, SMALLER_STEPS, sentence)


def led(lead: str, changes: int, advice: str, words: str) -> str:
    """*words* after a line of the app's: *lead*, the *changes* to the Library the run
    made, which stay, and *advice*."""
    made = count(changes, "change")
    kept = (
        [KEPT.format(changes=made, are=plural(changes, "is", "are"))] if changes else []
    )
    return f"{' '.join([lead, *kept, advice])}\n{words}"


def in_drive(message: dict[str, Any]) -> bool:
    """Whether *message* is one of an agent's drive's own: its turn, a cell's output or
    the engine's note on a turn with no <repl> block."""
    content = message.get("content")
    return (
        message.get("role") == "assistant"
        or message == NO_REPL_TURN
        or (isinstance(content, str) and OBSERVATION.match(content) is not None)
    )


def drive_start(messages: list[dict[str, Any]]) -> int:
    """Where the agent's drive begins in a call's *messages*: after its task, before
    which come its instructions, its worked examples and any conversation it carries on,
    whose cells' outputs may match the live ones."""
    at = len(messages)
    while at and in_drive(messages[at - 1]):
        at -= 1
    return at


def printed_lines(output: str) -> set[str]:
    return {line.strip() for line in output.splitlines()} - NOTHING_PRINTED


def repeats(output: str, printed: set[str]) -> bool:
    """Whether a cell's *output*, long enough to be worth reading again, is mostly the
    *printed* lines its drive printed before."""
    if len(output.strip()) < REREAD_CHARS:
        return False
    lines = printed_lines(output)
    return len(lines & printed) >= MOSTLY_PRINTED_BEFORE * len(lines)


def cut(output: str) -> str:
    """A cell's *output* past SENT_OUTPUT_CHARS as a call sends it: its start and end,
    half that each, around CUT_NOTE; each at a line break when its whole lines keep most
    of it, as a huge line (one-line JSON, a list's repr) would leave next to nothing."""
    half = SENT_OUTPUT_CHARS // 2
    head, tail = output[:half], output[-half:]
    lines_head, lines_tail = head.rpartition("\n")[0], tail.partition("\n")[2]
    # The line breaks either side of the cut, which the note's own line puts back.
    breaks = 0
    if len(lines_head) > WHOLE_LINES_KEEP * half:
        head, breaks = lines_head, breaks + 1
    if len(lines_tail) > WHOLE_LINES_KEEP * half:
        tail, breaks = lines_tail, breaks + 1
    note = CUT_NOTE.format(
        left=len(output) - len(head) - len(tail) - breaks,
        total=len(output),
        how="" if breaks == 2 else MID_LINE,
    )
    return f"{head}\n{note}\n{tail}"


def with_apps_notes(
    messages: list[dict[str, Any]], max_iter: int
) -> list[dict[str, Any]]:
    """*messages* as a chat call sends them: the engine's no-<repl> note as the app's,
    and, in the agent's drive, each cell's output past SENT_OUTPUT_CHARS cut(), then
    OWN_OUTPUT_NOTE after it, REPEATED_NOTE after one that raised nothing and is mostly
    what the drive was sent before, and TURNS_LEFT_NOTE after the turn that leaves it TURNS_LEFT_WARNED.
    Each message's notes depend only on those before it, so every call of a drive sends
    the same beginning as the last, which providers cache."""
    start = drive_start(messages)
    sent = [APPS_NO_REPL_TURN if m == NO_REPL_TURN else m for m in messages]
    printed: set[str] = set()
    turns = 0
    for at in range(start, len(messages)):
        message = messages[at]
        if message["role"] == "assistant":
            turns += 1
            continue
        notes = []
        if message != NO_REPL_TURN:
            output = output_of(message["content"])
            # Read whole: a cut may leave out the traceback.
            failed = raised_error(output) is not None
            if len(output) > SENT_OUTPUT_CHARS:
                output = cut(output)
                observation = f"<observation>\n{output}\n</observation>"
                sent[at] = message | {"content": observation}
            # What the model was sent: a cut-out line printed again is news to it.
            repeated = not failed and repeats(output, printed)
            notes = [OWN_OUTPUT_NOTE, REPEATED_NOTE] if repeated else [OWN_OUTPUT_NOTE]
            printed |= printed_lines(output)
        if turns == max_iter - TURNS_LEFT_WARNED:
            notes.append(TURNS_LEFT_NOTE)
        if notes:
            sent[at] = sent[at] | {"content": "\n".join([sent[at]["content"], *notes])}
    return sent


def tool_line(name: str, value: Any, description: str) -> str:
    """The line agents are shown for tool *name*, as the engine's func.describe words
    one, but by *name*: what they call, with its signature where Python reads one, or
    the name alone for an object they use through its methods. Each line of its
    *description* is indented under it, as one there reads as the system prompt's own."""
    if callable(value):
        try:
            name += str(inspect.signature(value))
        except (TypeError, ValueError):
            name += "(...)"
    body = f"\n{textwrap.indent(description, '  ')}" if description else ""
    return f"- `{name}`{body}"


def naming_tools() -> None:
    """Show agents each tool by the name they call it by. The engine names a tool by its
    Python value's __name__, so Func(count_words) granted as wordcount read as
    count_words(...), and an object, as the engine's own llm is, as fn(...); the value
    itself is left as it is."""
    describe = func.describe

    def described(self: func) -> str:
        if getattr(self.value, "__name__", None) == self.name:
            return describe(self)
        return tool_line(self.name, self.value, self.description)

    func.describe = described


def build_failure(error: ToolBuildError, config: V2Config) -> str:
    """Why a run couldn't start, as tools its *config* declares don't build, in words:
    for each, what went wrong, which namespaces grant it, and how to take it out."""
    granting = partial(granted_by, granting_registry(config))
    paragraphs = []
    for failure in error.failures:
        tool, detail = failure.tool, failure.detail
        # The run's copy of the tool's file is in its run folder: name the file alone.
        file = config.tools.get(tool, {}).get("factory_from")
        if file:
            folder = Path(file).parent
            for shown in (str(folder), os.path.relpath(folder)):
                detail = detail.replace(shown + os.sep, "")
        namespaces = granting(tool)
        where = (
            f"It is granted in {', '.join(namespaces)}: take it out of their tools"
            if namespaces
            else "Take it out of the tools of each namespace that grants it"
        )
        paragraphs.append(
            f"Tool {tool} doesn't build, so no chat can run while a namespace grants it:"
            f" {detail}\n{where} in the Library panel's Namespaces tab (Edit YAML)."
        )
    return "\n\n".join(paragraphs)


def granting_registry(config: V2Config) -> NamespaceRegistry | None:
    """*config*'s namespaces, to name those that grant a tool that doesn't build; None
    when they can't be read, so the failure is still worded, without them."""
    try:
        return build_namespace_registry(config)
    except Exception:
        log.exception("worker.granting_namespaces_unknown")
        return None


def granted_by(registry: NamespaceRegistry | None, tool: str) -> list[str]:
    """The namespaces of *registry* that grant *tool* themselves; none without one."""
    if registry is None:
        return []
    return granted_in(registry, in_chain_order(registry), tool)


@dataclass
class _Agent:
    drive: int
    cell: int | None = None  # its open cell
    code: str = ""  # the open cell's code
    answer: str | None = None  # the repr of the FinalAnswer this drive reached


class Recorder:
    """The structlog processor that turns deep_reasoner's run events into the app's, one
    JSON line each on *out*.

    deep_reasoner calls it on every thread a run logs from, so it holds one lock for its
    state and its writes. It returns every event dict unchanged and never raises. It
    writes nothing after the turn.end: sub-agents' threads may still log while the run
    closes, and the turn.end is the run's last event.
    """

    def __init__(
        self, out: TextIO, on_root_start: Callable[[], None], base_url: str
    ) -> None:
        self._out = out
        self._on_root_start = on_root_start
        self._base_url = base_url
        # chat calls started that logged no llm.call: cut off by a stop, or failed
        self._calls_going = 0
        self._lock = threading.RLock()
        self._agents: dict[int, _Agent] = {}
        # The worked examples left out of each agent started, until it says so.
        self._left_out: dict[int, tuple[dict[str, Any], ...]] = {}
        self._cells = 0
        self.changes_made = 0
        self._root: int | None = None
        self._ended = False
        self._handlers = {
            "agent.start": self._agent_start,
            "llm.call": self._llm_call,
            "claude.call": self._claude_call,
            "repl.execute": self._repl_execute,
            "agent.end": self._agent_end,
            "library.change": partial(
                self._tool_event, "library.change", CHANGE_FIELDS
            ),
            "page.open": partial(self._tool_event, "page.open", PAGE_FIELDS),
        }

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        name = event_dict.get("event")
        handler = self._handlers.get(name)
        if handler is None:
            return event_dict
        try:
            with self._lock:
                handler(event_dict)
        except Exception:
            log.exception("worker.event_not_recorded", name=name)
        return event_dict

    def left_out(self, node: int, examples: tuple[dict[str, Any], ...]) -> None:
        """The worked examples left out of agent *node*, for its agent.start to say."""
        with self._lock:
            self._left_out[node] = examples

    def call_started(self) -> None:
        with self._lock:
            self._calls_going += 1

    def end_turn(self, outcome: str, answer: str | None, detail: str | None) -> None:
        """The turn.end, with how many chat calls never finished, whose cost no one
        knows."""
        with self._lock:
            unfinished = self._calls_going
        self.emit(
            "turn.end",
            outcome=outcome,
            answer=answer,
            detail=detail,
            unfinished_calls=unfinished,
        )

    def emit(self, kind: str, **fields: Any) -> None:
        """Write one event; under the lock, so lines from different threads never mix."""
        with self._lock:
            if self._ended:
                return
            self._ended = kind == "turn.end"
            self._out.write(json.dumps({"kind": kind, **fields}, default=str) + "\n")
            self._out.flush()

    def _open_cell(self, node: int, source: str) -> int:
        self._cells += 1
        agent = self._agents[node]
        agent.cell, agent.code = self._cells, source
        self.emit("cell.start", node=node, cell=self._cells, code=source)
        return self._cells

    def _close_cell(self, node: int, source: str, output: str, status: str) -> None:
        agent = self._agents[node]
        self.emit(
            "cell.end",
            node=node,
            cell=agent.cell,
            code=source,
            output=output,
            status=status,
        )
        agent.cell, agent.code = None, ""

    def _agent_start(self, event: dict[str, Any]) -> None:
        node, parent = event["node_id"], event["parent"]
        if parent is None:
            self._root = node
            self._on_root_start()
        cell = None
        if parent in self._agents:
            # Under Claude Code or an opening turn, no cell of the parent's is known when
            # a sub-agent starts: an empty one opens for it to show in.
            cell = self._agents[parent].cell or self._open_cell(parent, "")
        drive = self._agents[node].drive + 1 if node in self._agents else 1
        self._agents[node] = _Agent(drive)
        self.emit(
            "agent.start",
            node=node,
            parent=parent,
            cell=cell,
            task=event["task"],
            namespace=event["namespace"],
            drive=drive,
        )
        for left_out in self._left_out.pop(node, ()):
            self.emit(LEFT_OUT, node=node, **left_out)

    def _llm_call(self, event: dict[str, Any]) -> None:
        self._calls_going -= 1
        node = event.get("agent_id") or self._root
        usage = event.get("usage") or {}
        own = self._own(event)
        self.emit(
            "usage",
            node=node,
            model=event["model"],
            own=own,
            tokens_in=usage.get("prompt_tokens") or 0,
            tokens_out=usage.get("completion_tokens") or 0,
            cost=cost_of(event["model"], usage, self._base_url),
        )
        if not own:
            return
        reply = event.get("response") or ""
        if thought := thought_of(reply):
            self.emit("think", node=node, text=thought)
        if (source := source_of(reply)) is not None:
            self._open_cell(node, source)

    def _claude_call(self, event: dict[str, Any]) -> None:
        node = event.get("agent_id") or self._root
        usage = event.get("usage") or {}
        cached = ("cache_read_input_tokens", "cache_creation_input_tokens")
        own = self._own(event)
        self.emit(
            "usage",
            node=node,
            model=event.get("model"),
            own=own,
            tokens_in=sum(usage.get(key) or 0 for key in ("input_tokens", *cached)),
            tokens_out=usage.get("output_tokens") or 0,
            cost=event.get("cost_usd"),
        )
        if own and (response := (event.get("response") or "").strip()):
            self.emit("think", node=node, text=response)

    def _repl_execute(self, event: dict[str, Any]) -> None:
        if not self._own(event):
            return
        node, source = event["agent_id"], event["source"]
        output = output_of(event["observation"])
        agent = self._agents[node]
        if agent.cell is None:
            self._open_cell(node, source)
        status = "done" if raised_error(output) is None else "failed"
        self._close_cell(node, source, output, status)
        if (answer := final_answer_of(output)) is not None:
            agent.answer = answer

    def _agent_end(self, event: dict[str, Any]) -> None:
        node, status = event["node_id"], event["status"]
        agent = self._agents.get(node)
        if agent is None:
            return
        if agent.cell is not None:
            cell_status = "stopped" if status == "stopped" else "failed"
            self._close_cell(node, agent.code, "", cell_status)
        result = None
        if status == "done" and agent.answer is not None:
            result = unquote(agent.answer)
        self.emit(
            "agent.end",
            node=node,
            status=status,
            result=result,
            detail=provider_led(event.get("detail")),
        )

    def _tool_event(
        self, kind: str, fields: Sequence[str], event: dict[str, Any]
    ) -> None:
        """What an app tool logged (a Library change, a page opened), in the open cell of
        the agent that called it."""
        node = event.get("agent_id") or self._root
        agent = self._agents.get(node)
        told = {key: event.get(key) for key in fields}
        cell = agent.cell if agent else None
        if kind == "library.change":
            self.changes_made += 1
        self.emit(kind, node=node, cell=cell, **told)

    def _own(self, event: dict[str, Any]) -> bool:
        """The event is a known agent's own model call or cell, not a tool's inside it."""
        node = event.get("node_id")
        return node == event.get("agent_id") and node in self._agents


def why_unrendered(error: Exception) -> str:
    """What failed as a worked example rendered, on one line: what Jinja or Python
    raised, unwrapped from the engine's error naming the example."""
    detail = detail_of(error.__cause__ or error)
    return detail.split("\n", 1)[0][:WHY_CHARS]


def where_stored(
    stored_in: dict[str, dict[str, str]], left_out: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    """Each worked example *left_out* with the namespace it is stored in, from, by
    *stored_in* (EXAMPLES_STORED_IN); None if it doesn't say."""
    return tuple(
        each | {"from": stored_in.get(each["namespace"], {}).get(each["name"])}
        for each in left_out
    )


def leaving_out_unrendered(
    recorder: Recorder, stored_in: dict[str, dict[str, str]]
) -> None:
    """Leave each worked example that fails to render out of its agent's prompt, all its
    turns, telling *recorder* which agent, once it started, and where the example is
    stored (*stored_in*), so the run goes on, where the engine would end it. A stop is
    no failure to render: it isn't an Exception."""
    rendered_turns = agent._rendered_turns
    start = agent.DeepReasonerAgentBase._start

    def rendered_or_none(
        decomposition: Any, render_ctx: dict[str, Any], namespace: str
    ) -> list[tuple[str, str]]:
        try:
            return rendered_turns(decomposition, render_ctx, namespace)
        # Logged as the run's event, which the chat shows; a log line would land in the
        # output of the cell a sub-agent starts in.
        except Exception as error:  # noqa: BLE001
            name = Decomposition.coerce(decomposition).name
            why = why_unrendered(error)
            left_out = {"name": name, "namespace": namespace, "why": why}
            LEFT_OUT_NOW.set((*LEFT_OUT_NOW.get(), left_out))
            return []

    def started(self: agent.DeepReasonerAgentBase) -> None:
        # Only an agent that started is told: one whose start failed never says so.
        token = LEFT_OUT_NOW.set(())
        try:
            start(self)
            if left_out := LEFT_OUT_NOW.get():
                stored = where_stored(stored_in, left_out)
                recorder.left_out(self._log_vars["node_id"], stored)
        finally:
            LEFT_OUT_NOW.reset(token)

    agent._rendered_turns = rendered_or_none
    agent.DeepReasonerAgentBase._start = started


def telling_last(line: str | None) -> None:
    """Tell *line* last, once, to every agent the engine gives its namespace's
    instructions (not one a cell starts with instructions= of its own): no namespace's
    suffix can, as an agent's is its chain's, root's first."""
    if not line:
        return
    rendered_instructions = agent._rendered_instructions

    def with_line(system_prompt: Any, suffix: str | None, render_ctx: dict) -> list:
        told = "\n\n".join(filter(None, [suffix, line]))
        return rendered_instructions(system_prompt, told, render_ctx)

    agent._rendered_instructions = with_line


def end_group() -> NoReturn:
    """SIGKILL the worker's process group, the worker included. Nothing it started may
    outlive it: a child a cell forked holds the events pipe, and so the run, open."""
    os.killpg(0, signal.SIGKILL)


def read_commands(stdin: int, stop: Callable[[], None]) -> NoReturn:
    """The server's commands on *stdin*, on a thread of their own, as they come. It is
    what ends the worker when the server goes, so a command that fails is logged and it
    reads on."""
    while True:
        try:
            select.select([stdin], [], [])
        except OSError:
            log.exception("worker.stdin_unreadable")
            end_group()
        try:
            take_commands(stdin, stop)
        except Exception:
            log.exception("worker.command_failed")


def take_commands(stdin: int, stop: Callable[[], None]) -> bool:
    """Act on the command the server has sent on *stdin*, if it has, without waiting:
    there is one, `stop`, sent at most once and whole. Whether it stopped."""
    with TAKING_COMMANDS:
        stopping = b"stop" in sent_so_far(stdin).split()
        if stopping:
            stop()
    return stopping


def sent_so_far(stdin: int) -> bytes:
    """What the server has sent on *stdin* and no one has read, without waiting. At EOF
    the server is gone, so nothing would read the events, and so when stdin can't be
    read: the worker ends its whole process group."""
    try:
        if not select.select([stdin], [], [], 0)[0]:
            return b""
        sent = os.read(stdin, STDIN_READ_SIZE)
    except OSError:
        log.exception("worker.stdin_unreadable")
        end_group()
    if not sent:
        end_group()
    return sent


def exit_after(seconds: float) -> None:
    time.sleep(seconds)
    log.warning("worker.close_run_hung", seconds=seconds)
    end_group()


def finish(reasoner: Any) -> NoReturn:
    """close_run under a watchdog, then end the group without waiting for sub-agents'
    threads."""
    threading.Thread(
        target=exit_after, args=(CLOSE_RUN_WATCHDOG_S,), daemon=True
    ).start()
    if reasoner is not None:
        try:
            close_run(reasoner)
        except Exception:
            log.exception("worker.close_run_failed")
    end_group()


def keep_app_logs_out_of_cells() -> None:
    """Send the app's own log records only where the engine's go, the worker's stderr,
    by the handler it has just set up: the engine copies every record logged in a cell,
    as by the library tool, into that cell's output, which the model and the tree read."""
    app_logs = logging.getLogger(APP_LOGGER)
    app_logs.handlers = list(logging.getLogger().handlers)
    app_logs.propagate = False


def counting_calls(recorder: Recorder, max_iter: int) -> None:
    """Tell *recorder* of each chat call the run starts: the engine logs only those that
    finish. Each call's messages go with any half of a character escaped: a cell's output
    may hold one, which the call couldn't send; and with the app's notes for agents of
    *max_iter* turns (with_apps_notes()), in what is sent only. A call the provider fails
    carries how long it waited (seconds_waited), for failure_detail(), and what that
    means as a note, which a parent's cell shows under a sub-agent's error."""
    create = AsyncCompletions.create

    async def counted(self: AsyncCompletions, *args: Any, **kwargs: Any) -> Any:
        # Before the count: a call escaped() raised on was never sent.
        if "messages" in kwargs:
            kwargs["messages"] = with_apps_notes(escaped(kwargs["messages"]), max_iter)
        recorder.call_started()
        started = time.monotonic()
        try:
            return await create(self, *args, **kwargs)
        except APIStatusError as error:
            error.seconds_waited = time.monotonic() - started
            if lead := provider_lead(error):
                error.add_note(lead)
            raise

    AsyncCompletions.create = counted


async def run(spec: dict[str, Any], folder: Path, out: TextIO) -> NoReturn:
    stop_asked = threading.Event()
    reasoner = None
    stdin = sys.stdin.fileno()

    def stop() -> None:
        stop_asked.set()
        if reasoner is not None:
            reasoner.stop()

    def stop_if_asked() -> None:
        # A stop does nothing before the root's first drive, so one asked earlier is
        # repeated here, when the root's drive has begun; one the stdin thread hasn't
        # read yet is taken here, before the root's first call.
        if not take_commands(stdin, stop) and stop_asked.is_set():
            reasoner.stop()

    base_url = spec["config"]["client"]["base_url"]
    recorder = Recorder(out, on_root_start=stop_if_asked, base_url=base_url)
    configure_structlog_fixture(console=False, extra_processors=[recorder])
    keep_app_logs_out_of_cells()
    leaving_out_unrendered(recorder, spec["config"].get(EXAMPLES_STORED_IN, {}))
    telling_last(spec["config"].get(TOLD_LAST))
    threading.Thread(target=read_commands, args=(stdin, stop), daemon=True).start()
    # A tool that doesn't build while the run builds the config's tools, at its start,
    # stops every run; one that doesn't later is that run's own failure.
    starting = True
    chats_folder = os.getcwd()
    try:
        config = V2Config.model_validate(spec["config"])
        counting_calls(recorder, config.max_iter)
        naming_tools()
        reasoner, alias = build_reasoner(config, run_dir=folder)
        # A tool's module may move the working folder as it loads: cells run in the chat's.
        os.chdir(chats_folder)
        starting = False
        earlier = spec["earlier"]
        if reasoner.backbone_type == CLAUDE_CODE:
            earlier = [for_claude(message) for message in earlier]
        # The conversation this run continues, as the agent's history= takes it: the
        # engine's build_reasoner takes no history=.
        reasoner._given_history = _parts(earlier, "history")
        with alias:
            answer = await reasoner.acall(spec["prompt"])
    except Exception as error:
        log.exception("worker.run_failed")
        outcome = "stopped" if stop_asked.is_set() else "failed"
        tools_failed = starting and isinstance(error, ToolBuildError)
        detail = (
            build_failure(error, config)
            if tools_failed
            else failure_detail(error, recorder.changes_made)
        )
        recorder.end_turn(outcome, None, detail)
        finish(reasoner)
    if reasoner.stopped:
        recorder.end_turn("stopped", None, None)
        finish(reasoner)
    text = answer if isinstance(answer, str) else repr(answer)
    if reasoner.exhausted:
        detail = exhausted_detail(recorder.changes_made, text)
        recorder.end_turn("exhausted", text, detail)
    else:
        recorder.end_turn("answered", text, None)
    finish(reasoner)


def main() -> None:
    sandbox_templates()
    folder = Path(sys.argv[1])
    # The cwd is the chat's folder: its own modules import, after every installed one.
    sys.path.append(os.getcwd())
    # The events keep the real stdout; anything else printed goes to stderr.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    spec = yaml.load((folder / SPEC).read_text(), Loader=YAML_LOADER)
    asyncio.run(run(spec, folder, out))


if __name__ == "__main__":
    main()
