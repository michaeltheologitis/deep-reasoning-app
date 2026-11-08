import ast
import asyncio
import errno
import hashlib
import os
import select
import shlex
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import aclosing, closing
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path

import pytest
import structlog
import yaml
from fake_model import FakeOpenAI, folded_answer, repl, scripted, task_and_turn
from structlog.testing import capture_logs

from deep_reasoning import runs as runs_module
from deep_reasoning.importing import import_config
from deep_reasoning.library import (
    default_namespace,
    engine_config,
    open_library,
)
from deep_reasoning.runs import (
    CHAT_DELETED,
    EARLIER_TURNS_NOTE,
    LOST,
    TOO_LONG,
    RunInProgress,
    Runs,
    Turn,
    earlier_messages,
    end_unfinished,
    new_run_id,
    read_events,
    read_turn,
    save_event,
)
from deep_reasoning.settings import VENDORS

log = structlog.get_logger()

pytestmark = pytest.mark.anyio

CHAT = "a-chat"
HELD_S = 30
FAN_OUT = {
    "Summarize two courses.": [
        repl(
            "r = run_all({c: anext(subagent().send(f'Summarize {c}.')) for c in ['C1', 'C2']})",
            "print(r)",
            think="Fan out.",
        ),
        repl("FinalAnswer(r)"),
    ],
    "Summarize C1.": [repl("print('C1 has 3 credits')"), repl("FinalAnswer('C1: 3')")],
    "Summarize C2.": [repl("FinalAnswer('C2: 4')", think="Easy.")],
}
FAN_OUT_ANSWER = "{'C1': 'C1: 3', 'C2': 'C2: 4'}"
PRINT_GROUP = repl("import os", "print(os.getpgrp())")
OVERFILL = 2**24  # bytes: more than a pipe holds
# More namespaces than a cell adds before a Stop at the first one's change ends it.
ADDS = 12
SAVE_S = 1.0
LEDGER = """\
import time
from pathlib import Path

from deep_reasoner.primitives import Func


class Ledger:
    \"\"\"A tool that takes a while to save itself when the run ends.\"\"\"

    def __init__(self, persist, seconds):
        self.persist_dir = Path(persist)
        self.seconds = seconds

    def save(self):
        time.sleep(self.seconds)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        (self.persist_dir / "saved").write_text("whole")


def build_ledger(client, tool_cfg):
    return Func(Ledger(tool_cfg["persist"], tool_cfg["seconds"]))
"""


CONNECT = sqlite3.connect
FULL = "database or disk is full"


def full_disk(room):
    """A SQLite connection whose disk has room for each run's first *room* events."""

    class FullDisk(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("INSERT INTO events") and parameters[1] > room:
                raise sqlite3.OperationalError(FULL)
            return super().execute(sql, parameters)

    return FullDisk


def saved(library, run):
    """The run's events as the app file holds them."""
    with closing(open_library(library)) as connection:
        return read_events(connection, run)


def turn_of(library, run):
    with closing(open_library(library)) as connection:
        return read_turn(connection, run)


async def run_to_the_end(runs, prompt, config, earlier=()):
    run = runs.start(CHAT, prompt, config, earlier)
    return run, [event async for event in runs.events(run)]


def of_kind(events, kind):
    return [event for event in events if event["kind"] == kind]


def group_of(events):
    """The worker's process group, which the run's first cell printed."""
    return int(of_kind(events, "cell.end")[0]["output"])


def slow_saving(folder):
    """Engine settings for a tool that takes SAVE_S to save itself as the run closes,
    and the file it writes once it has."""
    source = folder / "ledger.py"
    source.write_text(LEDGER)
    state = folder / "ledger"
    tool = {
        "factory": "build_ledger",
        "factory_from": str(source),
        "persist": str(state),
        "seconds": SAVE_S,
    }
    return {"tools": {"ledger": tool}}, state / "saved"


async def stop_at(runs, run, code):
    """Every event of the run, stopping it once a cell running *code* starts; and the
    seconds from the stop to the run's end."""
    events, stopped_at = [], None
    async for event in runs.events(run):
        events.append(event)
        if event["kind"] == "cell.start" and event["code"] == code:
            stopped_at = time.monotonic()
            assert runs.stop(run)
    return events, time.monotonic() - stopped_at


async def test_a_fan_out_streams_one_tree_in_order(home, library, fake_config):
    with FakeOpenAI(scripted(FAN_OUT)) as fake:
        run, events = await run_to_the_end(
            Runs(library, home), "Summarize two courses.", fake_config(fake)
        )
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["kind"] == "turn.start" and events[-1]["kind"] == "turn.end"
    starts = {event["node"]: event for event in of_kind(events, "agent.start")}
    fan_out = of_kind(events, "cell.start")[0]
    assert fan_out["node"] == 1
    assert [starts[node]["parent"] for node in (1, 2, 3)] == [None, 1, 1]
    assert starts[2]["cell"] == starts[3]["cell"] == fan_out["cell"]
    results = {event["node"]: event["result"] for event in of_kind(events, "agent.end")}
    assert results == {1: FAN_OUT_ANSWER, 2: "C1: 3", 3: "C2: 4"}
    fan_out_end = next(
        e for e in of_kind(events, "cell.end") if e["cell"] == fan_out["cell"]
    )
    child_ends = [e["seq"] for e in of_kind(events, "agent.end") if e["node"] != 1]
    assert fan_out["seq"] < starts[2]["seq"] and max(child_ends) < fan_out_end["seq"]
    assert events[-1]["outcome"] == "answered"
    assert events[-1]["answer"] == FAN_OUT_ANSWER
    assert saved(library, run) == events


async def test_a_cell_failed_only_when_it_raised(home, library, fake_config):
    recorded = 'Traceback (most recent call last):\n  File "<string>"\nKeyError: 3'
    plan = {
        "Check.": [
            repl("print('counting')", "{}[3]"),
            repl("raise ValueError('2 problems:\\ntools: bad\\nspawn: bad')"),
            repl(
                "import traceback",
                "try:",
                "    {}[3]",
                "except KeyError:",
                "    print(traceback.format_exc())",
                "print('carried on')",
            ),
            repl(f"print([{{'output': {recorded!r}, 'status': 'failed'}}])"),
            repl("FinalAnswer('checked')"),
        ]
    }
    with FakeOpenAI(scripted(plan)) as fake:
        _, events = await run_to_the_end(
            Runs(library, home), "Check.", fake_config(fake)
        )
    statuses = [event["status"] for event in of_kind(events, "cell.end")]
    assert statuses == ["failed", "failed", "done", "done", "done"]


async def test_a_run_works_in_its_folder_and_keeps_its_own_files_in_the_run_folder(
    home, library, fake_config, tmp_path
):
    folder = tmp_path / "thesis"
    folder.mkdir()
    write = repl(
        "import os, pathlib",
        "pathlib.Path('note.txt').write_text('written in the folder')",
        "print(os.getcwd())",
    )
    plan = {"Write a note.": [write, repl("FinalAnswer('noted')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Write a note.", fake_config(fake), [], folder=folder)
        events = [event async for event in runs.events(run)]
    assert events[-1]["outcome"] == "answered", events[-1]
    assert events[0]["folder"] == of_kind(events, "cell.end")[0]["output"]
    assert events[0]["folder"] == str(folder)
    assert [path.name for path in folder.iterdir()] == ["note.txt"]
    assert (folder / "note.txt").read_text() == "written in the folder"
    run_folder = {path.name for path in (home / "runs" / run).iterdir()}
    assert {"spec.yaml", "worker.log"} <= run_folder
    assert saved(library, run) == events


NAMED_OTHERWISE = '''\
from deep_reasoner.primitives import Func


def count_words(text: str) -> int:
    """How many words text has."""
    return len(text.split())


class Tally:
    """A running total: tally.add(k) adds k and returns the total."""

    def __init__(self):
        self.total = 0

    def add(self, k):
        self.total += k
        return self.total


def build_wordcount(client, params):
    return Func(count_words)


def build_tally(client, params):
    return Func(Tally())


def build_double(client, params):
    return Func(lambda x: 2 * x, description="Doubles x.")
'''


async def test_agents_are_shown_each_tool_by_the_name_they_call_it_by(
    home, library, fake_config, tmp_path
):
    source = tmp_path / "named.py"
    source.write_text(NAMED_OTHERWISE)
    tools = {
        name: {"factory": f"build_{name}", "factory_from": str(source)}
        for name in ("wordcount", "tally", "double")
    }
    use = repl("print(wordcount('a b c'), tally.add(2), double(4), wordcount.__name__)")
    plan = {"Use the tools.": [use, repl("FinalAnswer('used')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        config = fake_config(
            fake, tools=tools, namespaces={"root": {"tools": [*tools]}}
        )
        _, events = await run_to_the_end(Runs(library, home), "Use the tools.", config)
    assert events[-1]["outcome"] == "answered", events[-1]
    told = "\n".join(
        str(message["content"])
        for message in fake.calls[0].messages
        if message["role"] == "system"
    )
    assert "- `wordcount(text: str) -> int`\n  How many words text has." in told
    assert "- `tally`\n  A running total: tally.add(k)" in told
    assert "- `double(x)`\n  Doubles x." in told
    assert "count_words(" not in told and "<lambda>" not in told
    # What each tool is and does is left as it is.
    assert of_kind(events, "cell.end")[0]["output"] == "3 2 8 count_words"


async def test_a_tool_that_moves_the_working_folder_as_it_loads_leaves_cells_in_the_chats(
    home, library, fake_config, tmp_path
):
    folder, elsewhere = tmp_path / "thesis", tmp_path / "elsewhere"
    folder.mkdir()
    elsewhere.mkdir()
    source = tmp_path / "mover.py"
    source.write_text(
        f"import os\nfrom deep_reasoner.primitives import Func\n\n"
        f"os.chdir({str(elsewhere)!r})\n\n"
        "def build(client, params):\n    return Func(lambda: 1)\n"
    )
    write = repl(
        "import os, pathlib",
        "pathlib.Path('note.txt').write_text('here')",
        "print(os.getcwd())",
    )
    plan = {"Write a note.": [write, repl("FinalAnswer('noted')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        tools = {"mover": {"factory": "build", "factory_from": str(source)}}
        runs = Runs(library, home)
        config = fake_config(fake, tools=tools)
        run = runs.start(CHAT, "Write a note.", config, [], folder=folder)
        events = [event async for event in runs.events(run)]
    assert events[-1]["outcome"] == "answered", events[-1]
    assert of_kind(events, "cell.end")[0]["output"] == str(folder)
    assert [path.name for path in folder.iterdir()] == ["note.txt"]
    assert list(elsewhere.iterdir()) == []


async def test_the_folders_modules_import_but_never_shadow_installed_ones(
    home, library, fake_config, tmp_path
):
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "helpers.py").write_text("def greet():\n    return 'hi from helpers'\n")
    (folder / "json.py").write_text("raise SystemExit('the folder shadowed json')\n")
    look = repl(
        "import helpers, json, sys",
        "print(helpers.greet(), json.__file__ != sys.path[-1] + '/json.py')",
    )
    plan = {"Use my helpers.": [look, repl("FinalAnswer('used')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Use my helpers.", fake_config(fake), [], folder=folder)
        events = [event async for event in runs.events(run)]
    assert of_kind(events, "cell.end")[0]["output"] == "hi from helpers True"
    assert events[-1]["outcome"] == "answered", events[-1]


async def test_a_run_gets_its_vars_exactly_as_the_yaml_typed_them(
    home, library, fake_config
):
    look = repl("print(type(due).__name__, due, by_year, ascii(note))")
    plan = {"Read the vars.": [look, repl("FinalAnswer('read')")]}
    note = "café\x85中"  # NEL: written raw in YAML, it reads back as a space
    typed = {"due": date(2026, 10, 25), "by_year": {2024: 10}, "note": note}
    with FakeOpenAI(scripted(plan)) as fake:
        config = fake_config(fake, namespaces={"root": {"vars": typed}})
        _, events = await run_to_the_end(Runs(library, home), "Read the vars.", config)
    assert events[-1]["outcome"] == "answered", events[-1]
    output = of_kind(events, "cell.end")[0]["output"]
    assert output == "date 2026-10-25 {2024: 10} " + ascii(note)


async def test_stop_ends_every_agent_and_cell_stopped(home, library, fake_config):
    spin = repl("while True:", "    pass")
    plan = {
        "Keep a spinner going.": [PRINT_GROUP, repl("print(subagent('Spin.'))")],
        "Spin.": [spin],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Keep a spinner going.", fake_config(fake), [])
        events, seconds = await stop_at(runs, run, "while True:\n    pass")
    assert seconds < 2
    ends = {event["node"]: event["status"] for event in of_kind(events, "agent.end")}
    assert ends == {1: "stopped", 2: "stopped"}
    cell_ends = {
        event["cell"]: event["status"] for event in of_kind(events, "cell.end")
    }
    assert cell_ends == {1: "done", 2: "stopped", 3: "stopped"}
    assert events[-1]["outcome"] == "stopped"
    with pytest.raises(ProcessLookupError):
        os.killpg(group_of(events), 0)


async def test_stop_kills_a_cell_blocked_in_c(home, library, fake_config, tmp_path):
    unread = tmp_path / "unread"
    os.mkfifo(unread)
    reader = os.open(unread, os.O_RDONLY | os.O_NONBLOCK)
    # Written to a pipe nobody reads, so the write waits in C for good.
    write = f"os.write(os.open({str(unread)!r}, os.O_WRONLY), bytes({OVERFILL}))"
    plan = {"Wait a minute.": [PRINT_GROUP, repl("import os", write)]}
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home, stop_grace_s=0.5)
        run = runs.start(CHAT, "Wait a minute.", fake_config(fake), [])
        try:
            # Once the pipe holds what the cell wrote, the cell is in its write.
            wait = asyncio.to_thread(select.select, [reader], [], [], HELD_S)
            ready, _, _ = await wait
        finally:
            stopped_at = time.monotonic()
            stopped = runs.stop(run)
            events = [event async for event in runs.events(run)]
            seconds = time.monotonic() - stopped_at
            os.close(reader)
    assert ready, "the cell never reached its write"
    assert stopped and seconds < 3
    assert of_kind(events, "agent.end") == []
    assert events[-1]["outcome"] == "stopped"
    with pytest.raises(ProcessLookupError):
        os.killpg(group_of(events), 0)


async def test_after_a_stop_the_run_still_gets_the_grace_period_to_save(
    home, library, fake_config, tmp_path
):
    blocked_s, grace_s = 1.5, 2.0  # the stop lands 1.5 s in; saving would end at 2.5 s
    sleep = repl("import time", f"time.sleep({blocked_s})")
    settings, saved = slow_saving(tmp_path)
    with FakeOpenAI(scripted({"Wait a moment.": [sleep]})) as fake:
        runs = Runs(library, home, stop_grace_s=grace_s)
        config = fake_config(fake, **settings)
        run = runs.start(CHAT, "Wait a moment.", config, [])
        events, _ = await stop_at(runs, run, f"import time\ntime.sleep({blocked_s})")
    assert events[-1]["outcome"] == "stopped"
    assert saved.read_text() == "whole"


async def test_stop_before_the_root_starts_calls_no_model(home, library, fake_config):
    with FakeOpenAI(scripted(FAN_OUT)) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Summarize two courses.", fake_config(fake), [])
        assert runs.stop(run)
        events = [event async for event in runs.events(run)]
    assert fake.calls == []
    assert [event["status"] for event in of_kind(events, "agent.end")] == ["stopped"]
    assert events[-1]["outcome"] == "stopped" and events[-1]["cost"] == 0
    assert events[-1]["unfinished_calls"] == 0


def in_the_librarian(library, chat_config):
    """The config of a run of a chat in the librarian namespace, on *chat_config*'s model."""

    def config_for(folder):
        config = engine_config(
            library, "librarian", folder / "tools", CHAT, folder.name
        )
        chats = chat_config(folder)
        return config | {"model": chats["model"], "client": chats["client"]}

    return config_for


def namespaces_in(library):
    with closing(open_library(library)) as connection:
        return {name for (name,) in connection.execute("SELECT name FROM namespaces")}


async def test_a_stop_amid_library_writes_ends_the_run_with_each_written_change_recorded(
    home, library, fake_config
):
    names = [f"added{number}" for number in range(ADDS)]
    adds = repl(
        f"for name in {names!r}:",
        "    library.add_namespace(name, {'system_suffix': 'Be brief.'})",
    )
    before = namespaces_in(library)
    with FakeOpenAI(scripted({"Add them.": [adds]})) as fake:
        runs = Runs(library, home)
        config = in_the_librarian(library, fake_config(fake))
        run = runs.start(CHAT, "Add them.", config, [])
        events, stopped_at = [], None
        async for event in runs.events(run):
            events.append(event)
            if event["kind"] == "library.change" and stopped_at is None:
                stopped_at = time.monotonic()
                assert runs.stop(run)
    assert time.monotonic() - stopped_at < runs_module.STOP_GRACE_S
    assert [event["status"] for event in of_kind(events, "agent.end")] == ["stopped"]
    assert events[-1]["outcome"] == "stopped" and saved(library, run) == events
    assert events[-1]["unfinished_calls"] == 0
    logged = {event["name"] for event in of_kind(events, "library.change")}
    assert namespaces_in(library) - before == logged


async def test_a_run_whose_first_call_fails_ends_with_that_call_unfinished(
    home, library, fake_config
):
    def failing(messages):
        raise RuntimeError("the fake model is down")

    with FakeOpenAI(failing) as fake:
        _, events = await run_to_the_end(
            Runs(library, home), "Summarize two courses.", fake_config(fake)
        )
    assert of_kind(events, "usage") == []
    end = events[-1]
    assert (end["outcome"], end["cost"], end["unfinished_calls"]) == ("failed", 0, 1)


async def test_each_call_costs_what_its_server_says_it_cost(home, library, fake_config):
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.25}
    with FakeOpenAI(scripted(FAN_OUT), usage=lambda messages, reply: usage) as fake:
        _, events = await run_to_the_end(
            Runs(library, home), "Summarize two courses.", fake_config(fake)
        )
    costs = [event["cost"] for event in of_kind(events, "usage")]
    assert costs == [0.25] * len(fake.calls)
    assert events[-1]["cost"] == pytest.approx(0.25 * len(fake.calls))


async def test_a_vendors_model_on_a_server_of_your_own_costs_null_not_its_price(
    home, library, fake_config
):
    """A server that is no vendor and names no cost: a model under OpenAI's name there
    isn't OpenAI's to price."""
    with FakeOpenAI(scripted(FAN_OUT)) as fake:
        config = fake_config(fake, model="gpt-6-luna")
        _, events = await run_to_the_end(
            Runs(library, home), "Summarize two courses.", config
        )
    assert {event["cost"] for event in of_kind(events, "usage")} == {None}
    assert events[-1]["cost"] is None


async def test_a_follow_up_gets_each_earlier_message_and_answer_whole_as_turns(
    home, library, fake_config
):
    question = "Repeat this back: " + " ".join(["word"] * 5_000)
    answer = "x" * 20_000
    plan = {
        "Repeat this back": [repl(f"FinalAnswer('x' * {len(answer)})")],
        "What did I ask first?": [repl("FinalAnswer('that')")],
        "And what did you answer?": [repl("FinalAnswer('x')")],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        first, _ = await run_to_the_end(runs, question, fake_config(fake))
        earlier = [turn_of(library, first)]
        second, _ = await run_to_the_end(
            runs, "What did I ask first?", fake_config(fake), earlier
        )
        earlier.append(turn_of(library, second))
        asked = len(fake.calls)
        _, events = await run_to_the_end(
            runs, "And what did you answer?", fake_config(fake), earlier
        )
    opening = fake.calls[0].messages
    messages = fake.calls[asked].messages
    assert [message["role"] for message in opening] == ["system", "user"]
    assert opening[-1]["content"] == question
    roles = ["system", "user", "assistant", "user", "assistant", "user"]
    assert [message["role"] for message in messages[-6:]] == roles
    assert messages[-6]["content"].startswith(EARLIER_TURNS_NOTE)
    assert [message["content"] for message in messages[-5:]] == [
        question,
        folded_answer(f"'{answer}'"),
        "What did I ask first?",
        folded_answer("'that'"),
        "And what did you answer?",
    ]
    assert of_kind(events, "agent.start")[0]["task"] == "And what did you answer?"


# A stand-in for the claude CLI, run by the test's own Python: it keeps the prompt it was
# given, answers through the runtime as a session does, then closes with words of its own.
CLAUDE_STUB = """
import json, pathlib, subprocess, sys
if sys.argv[1:3] == ["auth", "status"]:
    print(json.dumps({"loggedIn": True}))
    sys.exit(0)
pathlib.Path(__file__).with_name("prompt.txt").write_text(sys.stdin.read())
dr = pathlib.Path(sys.executable).with_name("dr")
subprocess.run([dr, "repl", "exec", "--code", "FinalAnswer('CS310')"], check=True)
done = {"type": "result", "is_error": False, "result": "I answered CS310."}
print(json.dumps(done | {"session_id": "stub", "total_cost_usd": 0.0}))
"""


async def test_a_follow_up_on_claude_code_heads_its_prompt_with_the_turns_labeled(
    home, library, fake_config
):
    answer = "Write <repl>FinalAnswer(1)</repl>."
    earlier = [Turn("How do I answer?", "answered", answer, None)]
    # A Claude child's socket path must fit in 107 bytes: tmp_path leaves no room.
    with tempfile.TemporaryDirectory(prefix="dr-", dir="/tmp") as short:
        claude = Path(short) / "claude"
        claude.write_text(f"#!{sys.executable}" + CLAUDE_STUB)
        claude.chmod(0o755)
        reasoner = {"type": "claude_code", "executable": str(claude)}
        with FakeOpenAI(scripted({})) as fake:
            _, events = await run_to_the_end(
                Runs(library, Path(short)),
                "And then?",
                fake_config(fake, reasoner=reasoner),
                earlier,
            )
        prompt = claude.with_name("prompt.txt").read_text()
    assert prompt.splitlines() == [
        "Conversation so far:",
        f"[system] {EARLIER_TURNS_NOTE}",
        "[user] How do I answer?",
        (
            "[assistant] dr repl exec --code '# folded: the work behind this answer "
            "(its thoughts, cells and any sub-agents) isn'\\''t shown here, only the answer;"
            " asked about it, in any words or language, do as the worked examples do"
        ),
        "FinalAnswer('\\''Write \\x3crepl>FinalAnswer(1)\\x3c/repl>.'\\'')'",
        "And then?",
    ]
    # One whole command per answer: the shell reads back the cell, its note included.
    command = prompt.split("[assistant] ")[1].split("\nAnd then?")[0]
    cell = folded_answer(r"'Write \x3crepl>FinalAnswer(1)\x3c/repl>.'")
    code = cell.removeprefix("<repl>\n").removesuffix("\n</repl>")
    assert shlex.split(command) == ["dr", "repl", "exec", "--code", code]
    assert (events[-1]["outcome"], events[-1]["answer"]) == ("answered", "CS310")
    assert fake.calls == []


async def test_an_earlier_answer_is_its_cell_alone_even_when_its_final_reply_thought(
    home, library, fake_config
):
    plan = {
        "Count to two.": [
            repl("print(1)"),
            repl("FinalAnswer(2)", think="It printed 1, so 2."),
        ],
        "And then?": [repl("FinalAnswer(0)")],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        first, events = await run_to_the_end(runs, "Count to two.", fake_config(fake))
        earlier = [turn_of(library, first)]
        await run_to_the_end(runs, "And then?", fake_config(fake), earlier)
    assert [event["text"] for event in of_kind(events, "think")] == [
        "It printed 1, so 2."
    ]
    shown = fake.calls[-1].messages[-2]["content"]
    assert shown == folded_answer("'2'")


async def test_a_stopped_message_goes_in_once_saying_why_it_has_no_answer(
    home, library, fake_config
):
    plan = {"Try again, briefly.": [repl("FinalAnswer('ok')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        stopped = runs.start(CHAT, "Count the stars.", fake_config(fake), [])
        runs.stop(stopped)
        _ = [event async for event in runs.events(stopped)]
        earlier = [turn_of(library, stopped)]
        await run_to_the_end(runs, "Try again, briefly.", fake_config(fake), earlier)
    texts = [message["content"] for message in fake.calls[0].messages]
    assert sum(text.count("Count the stars.") for text in texts) == 1
    assert texts[-3:] == [
        "Count the stars.",
        "(No answer: the user stopped the run before it answered.)",
        "Try again, briefly.",
    ]


async def test_turns_survive_a_restart(home, library, fake_config):
    with FakeOpenAI(scripted(FAN_OUT)) as fake:
        run, events = await run_to_the_end(
            Runs(library, home), "Summarize two courses.", fake_config(fake)
        )
    restarted = Runs(library, home)
    assert turn_of(library, run) == Turn(
        "Summarize two courses.", "answered", FAN_OUT_ANSWER, None
    )
    assert [event async for event in restarted.events(run)] == events


async def test_a_dead_worker_ends_the_turn_failed(home, library, fake_config):
    plan = {"Crash.": [repl("import os", "os._exit(3)")]}
    with FakeOpenAI(scripted(plan)) as fake:
        run, events = await run_to_the_end(
            Runs(library, home), "Crash.", fake_config(fake)
        )
    assert events[-1]["outcome"] == "failed"
    assert "exited with code 3" in events[-1]["detail"]
    assert str(home / "runs" / run / "worker.log") in events[-1]["detail"]


async def test_a_child_a_cell_forked_ends_with_the_run(home, library, fake_config):
    # The child holds the events pipe: the run can end before its sleep only if it died.
    fork = repl(
        "import os, time",
        "if os.fork() == 0:",
        f"    time.sleep({HELD_S})",
        "    os._exit(0)",
    )
    plan = {"Fork.": [fork, repl("FinalAnswer('forked')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        started = time.monotonic()
        _, events = await run_to_the_end(
            Runs(library, home), "Fork.", fake_config(fake)
        )
    assert time.monotonic() - started < HELD_S
    assert events[-1]["outcome"] == "answered"


async def test_a_stream_rejoined_mid_run_replays_without_gaps(
    home, library, fake_config
):
    release = threading.Event()
    follow = scripted(FAN_OUT)

    def held_before_answering(messages):
        if task_and_turn(messages) == ("Summarize two courses.", 1):
            release.wait(HELD_S)
        return follow(messages)

    with FakeOpenAI(held_before_answering) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Summarize two courses.", fake_config(fake), [])
        stream = runs.events(run)
        seen = [await anext(stream) for _ in range(5)]
        rejoined = runs.events(run, after=3)
        replay = [await anext(rejoined)]
        release.set()
        seen += [event async for event in stream]
        replay += [event async for event in rejoined]
    assert seen[-1]["kind"] == "turn.end"
    assert replay == seen[3:]


async def test_a_chat_runs_one_message_at_a_time(home, library, fake_config):
    release = threading.Event()
    follow = scripted(FAN_OUT)

    def held(messages):
        release.wait(HELD_S)
        return follow(messages)

    with FakeOpenAI(held) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Summarize two courses.", fake_config(fake), [])
        with pytest.raises(RunInProgress):
            runs.start(CHAT, "And another thing.", fake_config(fake), [])
        assert runs.live(CHAT) == run
        release.set()
        _ = [event async for event in runs.events(run)]
    assert runs.live(CHAT) is None


async def until(runs, run, done):
    """The run's events up to the first one *done* accepts."""
    events = []
    async with aclosing(runs.events(run)) as stream:
        async for event in stream:
            events.append(event)
            if done(event):
                return events
    return events


async def test_a_chat_takes_its_next_message_once_its_turn_has_ended(
    home, library, fake_config, tmp_path
):
    settings, saved = slow_saving(tmp_path)
    plan = {"Hi.": [repl("FinalAnswer('hello')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        config = fake_config(fake, **settings)
        run = runs.start(CHAT, "Hi.", config, [])
        await until(runs, run, lambda event: event["kind"] == "turn.end")
        assert not saved.exists()  # the worker is still closing the run
        assert runs.live(CHAT) is None and runs.stop(run) is False
        _, events = await run_to_the_end(runs, "Hi.", config)
        _ = [event async for event in runs.events(run)]
    assert events[-1]["outcome"] == "answered"


async def test_remove_ends_the_chats_run_as_deleted_and_deletes_its_folder_once_ended(
    home, library, fake_config
):
    spin = "while True:\n    pass"
    with FakeOpenAI(scripted({"Spin.": [PRINT_GROUP, repl(spin)]})) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Spin.", fake_config(fake), [])
        watching = runs.events(run)
        await anext(watching)  # a stream open on the run, as a tab's is
        events = await until(runs, run, lambda event: event.get("code") == spin)
        await runs.remove(CHAT)
        rest = [event async for event in watching]
    with pytest.raises(ProcessLookupError):
        os.killpg(group_of(events), 0)
    assert rest[-1]["kind"] == "turn.end"
    assert (rest[-1]["outcome"], rest[-1]["detail"]) == ("deleted", CHAT_DELETED)
    assert not (home / "runs" / run).exists()
    assert saved(library, run) == [] and not runs.exists(run)


async def test_remove_waits_for_a_worker_still_closing_its_run(
    home, library, fake_config, tmp_path
):
    settings, saved = slow_saving(tmp_path)
    with FakeOpenAI(scripted({"Hi.": [repl("FinalAnswer('hello')")]})) as fake:
        runs = Runs(library, home)
        run = runs.start(CHAT, "Hi.", fake_config(fake, **settings), [])
        await until(runs, run, lambda event: event["kind"] == "turn.end")
        assert not saved.exists()  # the worker is still closing the run
        await runs.remove(CHAT)
    assert saved.read_text() == "whole"
    assert not (home / "runs" / run).exists()


async def test_remove_logs_a_folder_it_cannot_delete_and_deletes_the_rest(
    home, library, fake_config, monkeypatch
):
    with FakeOpenAI(scripted({"Hi.": [repl("FinalAnswer('hello')")]})) as fake:
        runs = Runs(library, home)
        stuck, _ = await run_to_the_end(runs, "Hi.", fake_config(fake))
        removed, _ = await run_to_the_end(runs, "Hi.", fake_config(fake))
    rmtree = shutil.rmtree

    def rmtree_refusing_stuck(path, *args, **kwargs):
        if Path(path).name == stuck:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", rmtree_refusing_stuck)
    with capture_logs() as logs:
        await runs.remove(CHAT)
    assert (home / "runs" / stuck).exists()
    assert not (home / "runs" / removed).exists()
    not_removed = [each for each in logs if each["event"] == "folders.not_removed"]
    assert [Path(each["folder"]).name for each in not_removed] == [stuck]


async def test_an_event_too_long_to_read_ends_the_turn_failed(
    home, library, fake_config, monkeypatch
):
    monkeypatch.setattr(runs_module, "LINE_LIMIT", 64 * 1024)
    plan = {"Shout.": [repl("print('x' * 100_000)")]}
    with FakeOpenAI(scripted(plan)) as fake:
        _, events = await run_to_the_end(
            Runs(library, home), "Shout.", fake_config(fake)
        )
    assert events[-1]["outcome"] == "failed"
    assert events[-1]["detail"] == TOO_LONG


async def test_long_outputs_and_answers_arrive_whole(home, library, fake_config):
    plan = {
        "Write a lot.": [
            repl("text = 'x' * 1_000_000", "print(text)"),
            repl("FinalAnswer(text)"),
        ]
    }
    with FakeOpenAI(scripted(plan)) as fake:
        _, events = await run_to_the_end(
            Runs(library, home), "Write a lot.", fake_config(fake)
        )
    assert of_kind(events, "cell.end")[0]["output"] == "x" * 1_000_000
    assert events[-1]["answer"] == "x" * 1_000_000


def test_a_run_the_app_stopped_under_is_ended_failed(library):
    run = "20261025-120000-000000"
    started = {
        "seq": 1,
        "t": 100.0,
        "kind": "turn.start",
        "chat": CHAT,
        "prompt": "Hi.",
    }
    used = {"seq": 2, "t": 101.5, "kind": "usage", "node": 1, "cost": 0.25}
    with closing(open_library(library)) as connection:
        save_event(connection, run, started)
        save_event(connection, run, used)
        assert end_unfinished(connection) == [run]
        assert end_unfinished(connection) == []
        last = read_events(connection, run)[-1]
    assert turn_of(library, run) == Turn("Hi.", "failed", None, LOST)
    assert (last["seq"], last["cost"], last["seconds"]) == (3, 0.25, 1.5)


async def test_an_event_the_app_file_refuses_ends_the_run_failed_and_frees_the_chat(
    home, library, fake_config, monkeypatch
):
    plan = {
        "Spin.": [repl("while True:", "    pass")],
        "Hi.": [repl("FinalAnswer('hello')")],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        monkeypatch.setattr(sqlite3, "connect", partial(CONNECT, factory=full_disk(2)))
        run, events = await run_to_the_end(runs, "Spin.", fake_config(fake))
        assert runs.live(CHAT) is None
        monkeypatch.setattr(sqlite3, "connect", CONNECT)
        _, again = await run_to_the_end(runs, "Hi.", fake_config(fake))
    assert events[-1]["kind"] == "turn.end" and events[-1]["outcome"] == "failed"
    assert FULL in events[-1]["detail"]
    assert saved(library, run) == events[:2]
    assert turn_of(library, run) == Turn("Spin.", "failed", None, LOST)
    assert again[-1]["outcome"] == "answered"


def test_run_ids_are_utc_so_they_sort_in_order_across_a_clock_change(monkeypatch):
    monkeypatch.setenv("TZ", "CET-1CEST,M3.5.0,M10.5.0/3")
    time.tzset()
    try:
        before = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        run = new_run_id()
        after = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    finally:
        monkeypatch.undo()
        time.tzset()
    assert before <= run <= after


def test_an_earlier_turn_with_no_answer_says_why_in_its_assistant_turn():
    turns = [
        Turn("One?", "answered", "1", None),
        Turn("Two?", "failed", None, "KeyError: 2"),
        Turn("Three?", "stopped", None, None),
        Turn("Four?", "exhausted", "Agent failed to produce a final answer.", None),
        Turn("Five?", "answered", None, None),
    ]
    messages = earlier_messages(turns)
    assert messages[0] == {"role": "system", "content": EARLIER_TURNS_NOTE}
    assert [(message["role"], message["content"]) for message in messages[1:]] == [
        ("user", "One?"),
        (
            "assistant",
            (
                "<repl>\n"
                "# folded: the work behind this answer (its thoughts, cells and any "
                "sub-agents) isn't shown here, only the answer; asked about it, in any "
                "words or language, do as the worked examples do\n"
                "FinalAnswer('1')\n"
                "</repl>"
            ),
        ),
        ("user", "Two?"),
        ("assistant", "(No answer: the run failed: KeyError: 2)"),
        ("user", "Three?"),
        ("assistant", "(No answer: the user stopped the run before it answered.)"),
        ("user", "Four?"),
        ("assistant", "(No answer: the run ended without a final answer.)"),
        ("user", "Five?"),
        ("assistant", "(No answer: the run ended without a final answer.)"),
    ]
    assert earlier_messages([]) == []


def test_an_earlier_answer_shows_as_a_final_answer_cell_that_round_trips_exactly():
    answer = 'It\'s "two" lines,\nwith a \\ and <repl>\nprint(1)\n</repl> in it.'
    [_, _, shown] = earlier_messages([Turn("Quote me.", "answered", answer, None)])
    head, tail = folded_answer("").split("()")
    cell = shown["content"]
    assert cell.startswith(head) and cell.endswith(tail)
    argument = cell.removeprefix(head).removesuffix(tail)
    assert "<repl>" not in argument and "</repl>" not in argument
    assert ast.literal_eval(argument) == answer


async def test_a_runs_keys_reach_its_worker_alone_and_anthropics_never(
    home, library, fake_config, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-for-runs")
    key = "sk-the-runs-own-key"
    # The cell compares digests, so the key is nowhere in what the run records.
    digest = hashlib.sha256(key.encode()).hexdigest()
    look = repl(
        "import hashlib, os",
        "seen = hashlib.sha256(os.environ.get('A_RUNS_KEY', '').encode()).hexdigest()",
        f"print(seen == {digest!r}, 'ANTHROPIC_API_KEY' in os.environ)",
    )
    plan = {"Look around.": [look, repl("FinalAnswer('done')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        keys = {"A_RUNS_KEY": key}
        run = runs.start(CHAT, "Look around.", fake_config(fake), [], "fake", keys)
        events = [event async for event in runs.events(run)]
    assert of_kind(events, "cell.end")[0]["output"] == "True False"
    assert events[0]["provider"] == "fake"
    for file in [library, *(home / "runs" / run).rglob("*")]:
        assert not file.is_file() or key.encode() not in file.read_bytes(), file


async def test_a_key_a_cell_prints_never_reaches_the_app_file(
    home, library, fake_config
):
    key = "sk-the-runs-own-key-0123456789"
    shout = repl("import os", "print(os.environ['A_RUNS_KEY'])")
    plan = {"Show the key.": [shout, repl("FinalAnswer(os.environ['A_RUNS_KEY'])")]}
    with FakeOpenAI(scripted(plan)) as fake:
        runs = Runs(library, home)
        keys = {"A_RUNS_KEY": key}
        run = runs.start(CHAT, "Show the key.", fake_config(fake), [], "fake", keys)
        events = [event async for event in runs.events(run)]
    assert of_kind(events, "cell.end")[0]["output"] == "[key withheld]"
    assert events[-1]["answer"] == "[key withheld]"
    assert saved(library, run) == events
    assert key.encode() not in library.read_bytes()


async def test_a_start_the_app_file_refuses_leaves_no_run_behind(
    home, library, fake_config, monkeypatch
):
    runs = Runs(library, home)
    monkeypatch.setattr(sqlite3, "connect", partial(CONNECT, factory=full_disk(0)))
    refused = pytest.raises(sqlite3.OperationalError, match=FULL)
    with FakeOpenAI(scripted(FAN_OUT)) as fake, refused:
        runs.start(CHAT, "Summarize two courses.", fake_config(fake), [])
    assert runs.live(CHAT) is None and fake.calls == []
    assert list((home / "runs").iterdir()) == []


def test_a_start_whose_config_fails_leaves_no_run_behind(home, library):
    def config_for(folder):
        (folder / "tools").mkdir()
        raise ValueError("the engine refuses the config")

    runs = Runs(library, home)
    with pytest.raises(ValueError, match="refuses"):
        runs.start(CHAT, "Hi.", config_for, [])
    assert runs.live(CHAT) is None
    assert list((home / "runs").iterdir()) == []


COURSES = {
    "CS": {
        "CS101": {"credits": 4, "year": 1},
        "CS102": {"credits": 4, "year": 1},
        "CS201": {"credits": 4, "year": 2},
    },
    "STAT": {
        "STAT101": {"credits": 3, "year": 1},
        "STAT201": {"credits": 3, "year": 2},
    },
}
FIRST_QUESTION = (
    "Which department is lighter for a first-year student, CS or STAT? Give each "
    "department to its own sub-agent to total its first-year credits from `courses`, "
    "then compare the totals."
)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
async def test_gpt_6_luna_on_openai_answers_and_remembers_the_chat(
    home, library, tmp_path
):
    """The app's part on a real model: a run that answers through sub-agents, every call
    priced, and a follow-up that is given the chat so far. Not the model's reasoning."""
    courses = tmp_path / "courses.yaml"
    courses.write_text(
        yaml.safe_dump({"namespaces": {"root": {"vars": {"courses": COURSES}}}})
    )
    with closing(open_library(library)) as connection:
        import_config(connection, courses)

    openai = VENDORS[0]
    chats_model = {
        "model": openai.first_model,
        "client": {"base_url": openai.base_url, "api_key_env": openai.key_env},
    }

    def config_for(folder):
        namespace = default_namespace(library)
        config = engine_config(library, namespace, folder / "tools", CHAT, folder.name)
        return config | chats_model

    runs = Runs(library, home)
    first, events = await run_to_the_end(runs, FIRST_QUESTION, config_for)
    assert events[-1]["outcome"] == "answered", events[-1]
    assert any(event["parent"] for event in of_kind(events, "agent.start"))
    costs = [event["cost"] for event in of_kind(events, "usage")]
    assert costs and None not in costs
    assert events[-1]["cost"] == pytest.approx(sum(costs))
    earlier = [turn_of(library, first)]
    _, follow_up = await run_to_the_end(
        runs, "Quote my first question word for word.", config_for, earlier
    )
    assert follow_up[-1]["outcome"] == "answered", follow_up[-1]
    assert "first-year" in follow_up[-1]["answer"]
    log.warning(
        "The live test spent real money",
        cost=events[-1]["cost"] + follow_up[-1]["cost"],
        first=events[-1]["answer"],
        calls=len(costs),
        agents=len(of_kind(events, "agent.start")),
        seconds=events[-1]["seconds"],
        follow_up=follow_up[-1]["answer"],
    )
