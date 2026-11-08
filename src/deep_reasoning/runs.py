"""Runs: each message is one run of deep_reasoner in a worker subprocess, streamed as events.

A run's events are in the app file, saved as they happen, so a copy of the file carries
every chat's past turns: the runs table names each run's chat, and the events table holds
each event as the JSON the page folds. The server numbers each event (seq, from 1) and
stamps its time (t). runs/<run>/ in the file's data folder holds the run's working files:
spec.yaml, what the worker runs, with the engine config; worker.log, the worker's stderr;
and what the engine writes. The model's key reaches the worker in its environment only,
never the run's folder or the file. The worker works in the chat's folder: its cwd, where
the agent's Python reads and writes.
"""

import asyncio
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import sys
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from itertools import takewhile
from pathlib import Path
from typing import Any, Literal

import jinja2
import structlog
import yaml

from deep_reasoning.folders import remove_unless_kept
from deep_reasoning.library import RUN_ID, RUN_ID_FORMAT, open_library, writing

log = structlog.get_logger()

Outcome = Literal["answered", "exhausted", "failed", "stopped", "deleted"]
RUNS = "runs"
SPEC = "spec.yaml"
# What a library.change event says of a change the library tool made to the Library.
CHANGE_FIELDS = ("what", "namespace", "name", "before", "after")
# What a page.open event says of a page the page tool saved.
PAGE_FIELDS = ("page", "title")
ADD_RUN = "INSERT INTO runs VALUES (?, ?)"
ADD_EVENT = "INSERT INTO events VALUES (?, ?, ?)"
FORWARD = "SELECT seq, data FROM events WHERE run = ? ORDER BY seq"
BACKWARD = "SELECT seq, data FROM events WHERE run = ? ORDER BY seq DESC"
# Runs and pages of no chat in the file; NOT IN would match none were a chat's id NULL.
ORPHANS = (
    "SELECT id FROM runs"
    " WHERE NOT EXISTS (SELECT 1 FROM chats WHERE chats.id = runs.chat)"
)
ORPHAN_PAGES = "NOT EXISTS (SELECT 1 FROM chats WHERE chats.id = pages.chat)"
# libyaml's where PyYAML has it: the pure-Python one takes ~0.6 s a MiB, on the loop.
YAML_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
STOP_GRACE_S = 5.0
MIB = 1024 * 1024
LINE_LIMIT = 64 * MIB
TOO_LONG = (
    f"The run sent an event longer than {LINE_LIMIT // MIB} MiB, more than the app "
    "reads as one event, so the app ended the run."
)
# Never handed to a run: with it set, Claude Code bills that key instead of the subscription.
WITHHELD = "ANTHROPIC_API_KEY"
SHUT_DOWN = "Deep Reasoning shut down while the run was going."
LOST = (
    "The run's end wasn't saved: the app stopped while it was going, or the Library file"
    " refused its events."
)
LOST_END = {"kind": "turn.end", "outcome": "failed", "answer": None, "detail": LOST}
NOT_SAVED = "The app couldn't save the run in the Library file, so it ended the run"
# A run whose chat was deleted while it went: only the streams still open on it see this.
CHAT_DELETED = "Its chat was deleted, so the run was stopped."
DELETED_END = {"outcome": "deleted", "answer": None, "detail": CHAT_DELETED}
KEY_WITHHELD = "[key withheld]"
# A shorter key is no secret (a local server's "x"), and withholding it would garble the run's text.
SHORTEST_SECRET = 8
# The largest number the page's JavaScript holds exactly: a shared file's events stay
# within it, so no sum or seq they make leaves what JSON and SQLite carry.
LARGEST_NUMBER = 2**53
# The server numbers a run's events from 1.
FIRST_SEQ = 1
# What an event row from a file someone else wrote may fail with.
UNREADABLE = (TypeError, ValueError, RecursionError, OverflowError)
NOT_ENDED = (*UNREADABLE, KeyError, IndexError, sqlite3.Error)
# JSON's escape of half a character (a surrogate). An emoji's pair loads as one character;
# a half alone, as a cell may print or a file from someone else hold, loads as text that
# can't be written as UTF-8, so no response or run's spec can carry it.
SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")
FINAL_ANSWER = "FinalAnswer: "
TRACEBACK_HEAD = re.compile(
    r"^(?:Traceback|  \+ Exception Group Traceback) \(most recent call last\):$",
    re.MULTILINE,
)
FRAME = "  "
EXCEPTION_LINE = re.compile(r"[^\s:]+(?:: .*)?")
# Each line of an exception group's traceback hangs off its frame: "  |", "  +-+", "    |".
GROUP_LINE = re.compile(r" +[|+]")
# A line of an exception group's own text, past its frame: "    | ValueError: 1".
GROUP_TEXT = re.compile(r" +\| (\S.*)")
# Where 0.3.0's REPL runs a cell: a traceback a cell raised starts here.
REPL_EXEC = re.compile(
    r'  File ".*[/\\]deep_reasoner[/\\].*", line \d+, in (?:run_code|_exec_and_capture)'
)
# What a follow-up's run is told ahead of the chat's earlier turns.
EARLIER_TURNS_NOTE = """\
This conversation began before this run: the user's earlier messages and your answers \
follow, then the user's next message. Any worked examples above are demonstrations, not \
this chat's messages: its first message is the first one after this note. The earlier \
runs have ended: their variables are gone, and their sub-agents can't be called again. \
Each earlier answer is shown only as the FinalAnswer it gave; how it was worked out (its \
thoughts, cells and sub-agents' steps) is kept, but in neither this conversation nor \
ctx. So a message about an earlier answer, in any language (how it was reached, whether \
it is right, what its code gave, what its sub-agents did or were told, or a page or \
table of it), is about that record: do as the worked examples do. Working it out again \
would be a new run, not the one that answered, so never answer such a message that way, \
or by guessing; work it out again only when the user asks for it to be redone."""
# An earlier turn's assistant turn: the cell that gave its answer, as the engine's worked
# examples show one, or why it has none. The cell says its work is folded away: shown the
# answer alone, root took a vague question about it ("did any of them have trouble?") as
# one it could answer itself, and guessed.
ANSWER = jinja2.Template(
    """\
{% if turn.outcome == "answered" and turn.answer is not none %}<repl>
# folded: the work behind this answer (its thoughts, cells and any sub-agents) isn't shown here, only the answer; asked about it, in any words or language, do as the worked examples do
FinalAnswer({{ answer }})
</repl>
{%- elif turn.outcome == "failed" %}(No answer: the run failed: {{ turn.detail }})
{%- elif turn.outcome == "stopped" %}(No answer: the user stopped the run before it answered.)
{%- else %}(No answer: the run ended without a final answer.){% endif %}""",
    undefined=jinja2.StrictUndefined,
)


class RunInProgress(Exception):
    """The chat already has a run going."""


@dataclass(frozen=True)
class Turn:
    message: str
    outcome: Outcome
    answer: str | None
    detail: str | None


@dataclass
class _Live:
    chat: str
    cwd: Path
    db: sqlite3.Connection  # the app file, which its events are saved in
    # for the worker's environment only, and in no repr a traceback might show
    keys: dict[str, str] = field(default_factory=dict, repr=False)
    # more variables of the worker's environment whose values are withheld as keys
    key_names: Sequence[str] = ()
    events: list[dict[str, Any]] = field(default_factory=list)
    watchers: set[asyncio.Queue] = field(default_factory=set)
    process: asyncio.subprocess.Process | None = None
    pump: asyncio.Task | None = None
    stop_asked: bool = False
    chat_deleted: bool = False
    kill_timer: asyncio.TimerHandle | None = None
    detail: str | None = None  # why the server ended it, if it did

    @property
    def ended(self) -> bool:
        """A turn.end is recorded; the worker may still be closing the run."""
        return self.events[-1]["kind"] == "turn.end"


def open_events(library: Path) -> sqlite3.Connection:
    """The app file at *library*, to save a run's events in as they happen, one commit
    each, on the event loop: in WAL mode with synchronous NORMAL, a commit waits on no
    fsync, and a power cut may lose the last few, never the file."""
    connection = open_library(library)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    except BaseException:
        connection.close()
        raise
    return connection


def save_event(connection: sqlite3.Connection, run: str, event: dict[str, Any]) -> None:
    """Add *event* to the run's in the app file; a turn.start adds the run to its chat."""
    with writing(connection):
        if event["kind"] == "turn.start":
            connection.execute(ADD_RUN, (run, event["chat"]))
        connection.execute(ADD_EVENT, (run, event["seq"], json.dumps(event)))


def in_range(parse: Callable[[str], int | float], text: str) -> int | float:
    """*text* as a number, by *parse*; refused past LARGEST_NUMBER, as NaN and infinity
    are, which no JSON response carries."""
    number = parse(text)
    if not abs(number) <= LARGEST_NUMBER:
        raise ValueError(f"{text[:20]} is out of range")
    return number


NUMBERS = {
    "parse_constant": partial(in_range, float),
    "parse_float": partial(in_range, float),
    "parse_int": partial(in_range, int),
}


def well_formed(event: Any, seq: Any) -> bool:
    """Whether *event*, read from the row numbered *seq*, is one the app could have
    written: an object with that seq, numbered from FIRST_SEQ, a time and a kind."""
    return (
        isinstance(event, dict)
        and isinstance(seq, int)
        and seq >= FIRST_SEQ
        and event.get("seq") == seq
        and isinstance(event.get("t"), int | float)
        and isinstance(event.get("kind"), str)
    )


def loaded_event(run: str, seq: Any, data: Any) -> dict[str, Any] | None:
    """The event in the run's row *seq*, or None for one the app could not have
    written, as a file from someone else may hold."""
    try:
        event = json.loads(data, **NUMBERS)
        # Bytes, which the app never writes, may hold a half unescaped.
        if type(data) is not str or SURROGATE_ESCAPE.search(data):
            json.dumps(event, ensure_ascii=False).encode()
    except UNREADABLE:
        event = None
    if well_formed(event, seq):
        return event
    log.warning("runs.event_left_out", run=run, seq=repr(seq))
    return None


def read_events(connection: sqlite3.Connection, run: str) -> list[dict[str, Any]]:
    """The run's events that load, in order."""
    rows = connection.execute(FORWARD, (run,))
    events = (loaded_event(run, seq, data) for seq, data in rows)
    return [event for event in events if event is not None]


def first_loaded(
    connection: sqlite3.Connection, run: str, query: str
) -> dict[str, Any] | None:
    """The run's first event that loads, in the order of *query*: FORWARD or BACKWARD."""
    with closing(connection.execute(query, (run,))) as rows:
        events = (loaded_event(run, seq, data) for seq, data in rows)
        return next((event for event in events if event is not None), None)


def started(event: dict[str, Any] | None) -> bool:
    """Whether a run's first event that loads is its turn.start, with its message: else,
    as in a file from someone else, the run is left out of the page and of follow-ups."""
    return (
        event is not None
        and event["kind"] == "turn.start"
        and isinstance(event.get("prompt"), str)
    )


def has_ended(event: dict[str, Any] | None) -> bool:
    """Whether a run's last event that loads is its turn.end."""
    return event is not None and event["kind"] == "turn.end"


def total_cost(costs: Sequence[Any]) -> float | None:
    """What *costs* add up to; None when any of them is unknown."""
    if all(isinstance(cost, int | float) for cost in costs):
        return math.fsum(costs)
    return None


def totals(events: Sequence[dict[str, Any]], until: float) -> dict[str, Any]:
    """A turn's cost, None when any call's is unknown, and its seconds until *until*."""
    costs = [event.get("cost") for event in events if event["kind"] == "usage"]
    return {"cost": total_cost(costs), "seconds": until - events[0]["t"]}


def lost_end(events: Sequence[dict[str, Any]], seq: int) -> dict[str, Any]:
    """The turn.end, numbered *seq*, of a run that has none, as the app stopped under it:
    failed, as of its last event."""
    until = events[-1]["t"]
    return {"seq": seq, "t": until} | LOST_END | totals(events, until)


def ended(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A run's events, ending with a turn.end: one with none, as one the app stopped
    under or couldn't save the end of, ends failed, though not in the file."""
    if not events or has_ended(events[-1]):
        return events
    return [*events, lost_end(events, events[-1]["seq"] + 1)]


def read_turn(connection: sqlite3.Connection, run: str) -> Turn | None:
    """An ended run's message and how it ended, from its first and last events that load;
    None for a run that didn't start, as one in a file from someone else may not have."""
    start = first_loaded(connection, run, FORWARD)
    if not started(start):
        log.warning("runs.turn_left_out", run=run)
        return None
    end = first_loaded(connection, run, BACKWARD)
    if not has_ended(end):
        end = LOST_END
    return Turn(
        start["prompt"], end.get("outcome"), end.get("answer"), end.get("detail")
    )


def chat_runs(connection: sqlite3.Connection, chat: str) -> list[str]:
    """The chat's runs, oldest first. A run's id names its folder, so one the app could
    not have made, as a file from someone else may hold, is left out."""
    rows = connection.execute("SELECT id FROM runs WHERE chat = ? ORDER BY id", (chat,))
    runs = []
    for (run,) in rows:
        if isinstance(run, str) and re.fullmatch(RUN_ID, run):
            runs.append(run)
        else:
            log.warning("runs.left_out", run=repr(run))
    return runs


def chat_turns(
    connection: sqlite3.Connection, chat: str, going: str | None = None
) -> list[dict[str, Any]]:
    """The chat's turns as the page reads them: each run's events, oldest run first. A run
    that isn't *going* and has no turn.end, as one whose end the app couldn't save, ends
    failed; one with no turn.start, as a file from someone else may hold, is left out."""
    turns = []
    for run in chat_runs(connection, chat):
        events = read_events(connection, run)
        if not started(events[0] if events else None):
            log.warning("runs.turn_left_out", run=run)
            continue
        turns.append({"run": run, "events": events if run == going else ended(events)})
    return turns


def delete_runs(connection: sqlite3.Connection, chat: str) -> list[str]:
    """Delete the chat's runs, their events and the pages they made from the app file;
    the runs' ids."""
    with writing(connection):
        runs = chat_runs(connection, chat)
        connection.execute(
            "DELETE FROM events WHERE run IN (SELECT id FROM runs WHERE chat = ?)",
            (chat,),
        )
        connection.execute("DELETE FROM runs WHERE chat = ?", (chat,))
        connection.execute("DELETE FROM pages WHERE chat = ?", (chat,))
    return runs


def delete_orphans(connection: sqlite3.Connection) -> None:
    """Delete the runs, their events and pages, of chats no longer in the file, as when
    the app stopped between deleting a chat and its runs; a file it can't write keeps
    them."""
    try:
        with writing(connection):
            connection.execute(f"DELETE FROM events WHERE run IN ({ORPHANS})")
            connection.execute(f"DELETE FROM runs WHERE id IN ({ORPHANS})")
            connection.execute(f"DELETE FROM pages WHERE {ORPHAN_PAGES}")
    except sqlite3.Error as error:
        log.warning("runs.orphans_kept", error=str(error))


def end_unfinished(connection: sqlite3.Connection) -> list[str]:
    """End, as failed, every run the app stopped under without a turn.end; their ids."""
    runs = [run for (run,) in connection.execute("SELECT id FROM runs ORDER BY id")]
    return [run for run in runs if ended_as_lost(connection, run)]


def ended_as_lost(connection: sqlite3.Connection, run: str) -> bool:
    """End *run* as failed, after its highest whole seq, unless its last event that loads
    is its turn.end; whether it did. One it can't end, as a file from someone else or one
    it can't write may hold, is left as it is."""
    try:
        if has_ended(first_loaded(connection, run, BACKWARD)):
            return False
        events = read_events(connection, run)
        top = connection.execute(
            "SELECT max(seq) FROM events WHERE run = ? AND typeof(seq) = 'integer'",
            (run,),
        ).fetchone()[0]
        save_event(connection, run, lost_end(events, top + 1))
    except NOT_ENDED as error:
        log.warning("runs.not_ended", run=run, error=repr(error))
        return False
    return True


def raised_error(output: str) -> str | None:
    """What a cell raised, as its traceback's exception lines, or None when it raised
    nothing. repl.execute logs no sign of it but the observation, which the engine ends
    with the traceback, in format_exc's shape, then at most a FinalAnswer line. A raised
    traceback starts at the REPL's exec, and its message may run on over lines; one the
    cell printed starts in the cell, so it must end at its exception line. A printed repr,
    such as a recorded step's, escapes its newlines."""
    body, _, last = output.rpartition("\n")
    if not last.startswith(FINAL_ANSWER):
        body = output
    heads = list(TRACEBACK_HEAD.finditer(body))
    if not heads:
        return None
    head, *lines = body[heads[-1].start() :].rstrip("\n").split("\n")
    if head.startswith(" "):
        return group_error(lines)
    frames = list(takewhile(lambda line: line.startswith(FRAME), lines))
    rest = lines[len(frames) :]
    if not frames or not rest or not EXCEPTION_LINE.fullmatch(rest[0]):
        return None
    if len(rest) > 1 and REPL_EXEC.fullmatch(frames[0]) is None:
        return None
    return "\n".join(rest)


def group_error(lines: list[str]) -> str | None:
    """An exception group's exception lines, its own then its sub-exceptions', from the
    lines under its traceback's head; None unless each hangs off its frame."""
    if not lines or not all(GROUP_LINE.match(line) for line in lines):
        return None
    texts = [match[1] for line in lines if (match := GROUP_TEXT.fullmatch(line))]
    return "\n".join(text for text in texts if EXCEPTION_LINE.fullmatch(text))


def secrets_of(env: dict[str, str], names: Sequence[str]) -> list[str]:
    """The values of *names* in the worker's *env* long enough to be secrets, longest
    first, so a key that holds another is withheld whole."""
    values = {env[name] for name in names if len(env.get(name, "")) >= SHORTEST_SECRET}
    return sorted(values, key=len, reverse=True)


def withheld(value: Any, secrets: Sequence[str]) -> Any:
    """*value* with each of *secrets* replaced in all its text, all the way down."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, KEY_WITHHELD)
        return value
    if isinstance(value, dict):
        return {name: withheld(each, secrets) for name, each in value.items()}
    if isinstance(value, list):
        return [withheld(each, secrets) for each in value]
    return value


def answer_literal(answer: Any) -> str:
    """*answer* as a Python literal, with each <repl> tag's < written as an escape, so a
    Claude session's rendering of the cell, which ends at the first </repl>, keeps it
    whole."""
    return (
        repr(answer).replace("<repl>", "\\x3crepl>").replace("</repl>", "\\x3c/repl>")
    )


def earlier_messages(turns: Sequence[Turn]) -> list[dict[str, str]]:
    """The chat so far as chat messages for a fresh run: a system note, then each earlier
    message, whole, as the user's turn and its answer, whole, as the assistant's, in the
    cell that gave it. Nothing for the chat's first message."""
    if not turns:
        return []
    messages = [{"role": "system", "content": EARLIER_TURNS_NOTE}]
    for turn in turns:
        messages.append({"role": "user", "content": turn.message})
        answer = ANSWER.render(turn=turn, answer=answer_literal(turn.answer))
        messages.append({"role": "assistant", "content": answer})
    return messages


def new_run_id() -> str:
    """The start time in UTC, to the microsecond: run ids sort in the order the runs
    began, across a change of the local clock."""
    return datetime.now(UTC).strftime(RUN_ID_FORMAT)


def worker_env(keys: dict[str, str]) -> dict[str, str]:
    """The app's environment with the run's *keys* added, and never ANTHROPIC_API_KEY."""
    return {
        name: value for name, value in (os.environ | keys).items() if name != WITHHELD
    }


def kill_group(pid: int) -> None:
    """SIGKILL the process group the worker leads: the worker and all it started."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


class Runs:
    """The runs of the app file at *library*, with their folders in *home*, its data
    folder; and the live ones: each one's worker, its events so far, and the streams
    watching it."""

    def __init__(
        self, library: Path, home: Path, *, stop_grace_s: float = STOP_GRACE_S
    ) -> None:
        self.library = library
        self.home = home
        self._stop_grace_s = stop_grace_s
        self._live: dict[str, _Live] = {}

    def live(self, chat: str) -> str | None:
        """The chat's run that is going, if one is: one whose turn has not ended."""
        return next(
            (
                run
                for run, live in self._live.items()
                if live.chat == chat and not live.ended
            ),
            None,
        )

    def exists(self, run: str) -> bool:
        with closing(open_library(self.library)) as connection:
            found = connection.execute("SELECT 1 FROM runs WHERE id = ?", (run,))
            return found.fetchone() is not None

    def start(
        self,
        chat: str,
        prompt: str,
        config_for: Callable[[Path], dict[str, Any]],
        earlier: Sequence[Turn],
        provider: str | None = None,
        keys: dict[str, str] | None = None,
        folder: Path | None = None,
        key_names: Sequence[str] = (),
    ) -> str:
        """Start a run of *prompt* in *chat*, given the chat's *earlier* turns, and return
        its id. *config_for* gives the engine config for the run's folder, which the
        run's id names; *provider* names who serves its model, and *keys* go into the
        worker's environment only. The values of *keys* and of *key_names* in that
        environment are withheld from the run's events. The worker works in *folder*,
        the run's own folder unless given. Raises RunInProgress while the chat has a
        run going."""
        if self.live(chat) is not None:
            raise RunInProgress(chat)
        run = new_run_id()
        run_folder = self.home / RUNS / run
        run_folder.mkdir(parents=True)
        db = None
        # A start that fails leaves nothing behind, and as the run is listed only once its
        # turn.start is saved, holds up no chat.
        try:
            config = config_for(run_folder)
            namespace = config.get("entry_namespace")
            spec = {
                "chat": chat,
                "prompt": prompt,
                "earlier": earlier_messages(earlier),
                "config": config,
            }
            # YAML keeps dates and int keys; ASCII, as a raw NEL would read back as a space.
            spec_yaml = yaml.dump(spec, Dumper=YAML_DUMPER, sort_keys=False)
            (run_folder / SPEC).write_text(spec_yaml)
            db = open_events(self.library)
            live = _Live(chat, folder or run_folder, db, keys or {}, key_names)
            start = {
                "kind": "turn.start",
                "run": run,
                "chat": chat,
                "prompt": prompt,
                "namespace": namespace,
                "model": config.get("model"),
                "provider": provider,
                "folder": str(live.cwd),
            }
            self._record(run, live, start)
        except BaseException:
            if db is not None:
                db.close()
            shutil.rmtree(run_folder, ignore_errors=True)
            raise
        self._live[run] = live
        live.pump = asyncio.create_task(self._pump(run, live))
        return run

    def stop(self, run: str) -> bool:
        """Ask a run to stop: its worker stops the agent tree and ends the turn, or the
        group is killed after the grace period. False when the run is not going."""
        live = self._live.get(run)
        if live is None or live.ended:
            return False
        if not live.stop_asked:
            live.stop_asked = True
            self._ask_to_stop(live)
        return True

    async def events(self, run: str, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        """The run's events after seq *after*; for a run that is going, then each new one
        as it happens, until the run is over."""
        live = self._live.get(run)
        if live is None:
            with closing(open_library(self.library)) as connection:
                events = ended(read_events(connection, run))
            for event in events:
                if event["seq"] > after:
                    yield event
            return
        # The backlog and the subscription are taken in one step on the event loop, so
        # no event is missed or repeated between them.
        queue = asyncio.Queue()
        backlog = live.events[after:]
        live.watchers.add(queue)
        try:
            for event in backlog:
                yield event
            while (event := await queue.get()) is not None:
                yield event
        finally:
            live.watchers.discard(queue)

    def folders_in_use(self, besides: str) -> list[str]:
        """The folders the runs of every chat but *besides* are working in."""
        return [str(live.cwd) for live in self._live.values() if live.chat != besides]

    async def remove(self, chat: str, keep: Sequence[str] = ()) -> None:
        """Delete the chat's runs, their events and their folders, stopping the one going,
        once each of their workers has exited: a worker writes into its run's folder until
        then, even after turn.end. A run's folder that is or holds a folder in *keep* is
        kept."""
        workers = []
        for run, live in self._live.items():
            if live.chat == chat:
                live.chat_deleted = True
                self.stop(run)
                workers.append(live.pump)
        if workers:
            await asyncio.wait(workers)
        with closing(open_library(self.library)) as connection:
            runs = delete_runs(connection, chat)
        for run in runs:
            remove_unless_kept(self.home / RUNS / run, keep)

    async def close(self) -> None:
        """End every run that is going: the app is shutting down."""
        lives = list(self._live.values())
        for live in lives:
            live.detail = SHUT_DOWN
            if live.process is None:
                live.pump.cancel()
            else:
                kill_group(live.process.pid)
        await asyncio.gather(*(live.pump for live in lives), return_exceptions=True)

    def _ask_to_stop(self, live: _Live) -> None:
        process = live.process
        if process is None or process.returncode is not None:
            return
        process.stdin.write(b"stop\n")
        self._kill_after_grace(live)

    def _kill_after_grace(self, live: _Live) -> None:
        """Kill the worker's group after the grace period, unless it exits first."""
        if live.kill_timer is not None:
            live.kill_timer.cancel()
        loop = asyncio.get_running_loop()
        live.kill_timer = loop.call_later(
            self._stop_grace_s, kill_group, live.process.pid
        )

    def _record(self, run: str, live: _Live, event: dict[str, Any]) -> None:
        """Number and stamp *event*, send it to the run's streams, and save it. Sent
        first: a run whose event can't be saved ends, and its streams still see it end."""
        event = {"seq": len(live.events) + FIRST_SEQ, "t": time.time()} | event
        if event["kind"] == "turn.end":
            event |= totals(live.events, event["t"])
            if live.chat_deleted:
                event |= DELETED_END
        live.events.append(event)
        for queue in live.watchers:
            queue.put_nowait(event)
        save_event(live.db, run, event)

    async def _pump(self, run: str, live: _Live) -> None:
        """Start the worker and record its events until it exits; then end the run."""
        folder = self.home / RUNS / run
        try:
            with (folder / "worker.log").open("ab") as worker_log:
                # -P: a module in the chat's folder mustn't shadow the worker's own
                live.process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-P",
                    "-m",
                    "deep_reasoning.worker",
                    str(folder),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=worker_log,
                    cwd=live.cwd,
                    env=worker_env(live.keys),
                    start_new_session=True,
                    limit=LINE_LIMIT,
                )
            if live.stop_asked:
                self._ask_to_stop(live)
            await self._read(run, live)
        except sqlite3.Error as error:
            log.exception("runs.not_saved", run=run)
            live.detail = f"{NOT_SAVED}: {error}."
        except Exception as error:
            log.exception("runs.lost_the_worker", run=run)
            live.detail = f"The app lost the run's worker: {error}"
        finally:
            await self._end(run, live)

    async def _read(self, run: str, live: _Live) -> None:
        """Record each event the worker writes, with the keys in its environment withheld:
        what a cell prints goes into the file, which is shared."""
        names = [*live.keys, *live.key_names]
        secrets = secrets_of(worker_env(live.keys), names)
        while True:
            try:
                line = await live.process.stdout.readline()
            except ValueError:
                live.detail = TOO_LONG
                return
            if not line:
                return
            self._record(run, live, withheld(json.loads(line), secrets))
            if live.ended:
                # The worker saves the run's tools as it closes: it gets the whole grace
                # period for that, even after a Stop that used up most of it.
                self._kill_after_grace(live)

    async def _end(self, run: str, live: _Live) -> None:
        """Kill what is left of the worker's group, wait for the worker, and end the
        turn if the worker didn't. The chat and the streams are let go whatever fails."""
        try:
            if live.process is not None:
                kill_group(live.process.pid)
                await live.process.wait()
            if live.kill_timer is not None:
                live.kill_timer.cancel()
            if not live.ended:
                self._record(run, live, self._turn_end(run, live))
        except sqlite3.Error:
            # Its streams saw the turn.end; the file gets one when the app next starts.
            log.exception("runs.end_not_saved", run=run)
        finally:
            live.db.close()
            del self._live[run]
            for queue in live.watchers:
                queue.put_nowait(None)

    def _turn_end(self, run: str, live: _Live) -> dict[str, Any]:
        """The turn.end of a run whose worker ended without writing one."""
        end = {"kind": "turn.end", "answer": None}
        if live.stop_asked:
            return end | {"outcome": "stopped", "detail": None}
        detail = live.detail
        if detail is None:
            code = live.process.returncode if live.process else None
            worker_log = self.home / RUNS / run / "worker.log"
            detail = (
                f"The run's worker exited with code {code} before the run ended; "
                f"its log is {worker_log}."
            )
        return end | {"outcome": "failed", "detail": detail}
