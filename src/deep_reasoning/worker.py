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
from openai.resources.chat.completions import AsyncCompletions

from deep_reasoning.app_tools import CLAUDE_CODE
from deep_reasoning.library import granted_in, in_chain_order, sandbox_templates
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


def tool_line(name: str, value: Any, description: str) -> str:
    """The line agents are shown for tool *name*, as the engine's func.describe words
    one, but by *name*: what they call, with its signature where Python reads one, or
    the name alone for an object they use through its methods."""
    if callable(value):
        try:
            name += str(inspect.signature(value))
        except (TypeError, ValueError):
            name += "(...)"
    body = f"\n  {description}" if description else ""
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
            detail=event.get("detail"),
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


def leaving_out_unrendered(recorder: Recorder) -> None:
    """Leave each worked example that fails to render out of its agent's prompt, all its
    turns, telling *recorder* which agent, once it started, so the run goes on, where
    the engine would end it. A stop is no failure to render: it isn't an Exception."""
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
                recorder.left_out(self._log_vars["node_id"], left_out)
        finally:
            LEFT_OUT_NOW.reset(token)

    agent._rendered_turns = rendered_or_none
    agent.DeepReasonerAgentBase._start = started


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


def counting_calls(recorder: Recorder) -> None:
    """Tell *recorder* of each chat call the run starts: the engine logs only those that
    finish. Each call's messages go with any half of a character escaped: a cell's output
    may hold one, which the call couldn't send."""
    create = AsyncCompletions.create

    async def counted(self: AsyncCompletions, *args: Any, **kwargs: Any) -> Any:
        # Before the count: a call escaped() raised on was never sent.
        if "messages" in kwargs:
            kwargs["messages"] = escaped(kwargs["messages"])
        recorder.call_started()
        return await create(self, *args, **kwargs)

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
    counting_calls(recorder)
    leaving_out_unrendered(recorder)
    threading.Thread(target=read_commands, args=(stdin, stop), daemon=True).start()
    # A tool that doesn't build while the run builds the config's tools, at its start,
    # stops every run; one that doesn't later is that run's own failure.
    starting = True
    chats_folder = os.getcwd()
    try:
        config = V2Config.model_validate(spec["config"])
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
        detail = build_failure(error, config) if tools_failed else detail_of(error)
        recorder.end_turn(outcome, None, detail)
        finish(reasoner)
    if reasoner.stopped:
        recorder.end_turn("stopped", None, None)
        finish(reasoner)
    outcome = "exhausted" if reasoner.exhausted else "answered"
    text = answer if isinstance(answer, str) else repr(answer)
    recorder.end_turn(outcome, text, None)
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
