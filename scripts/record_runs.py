"""Record real runs against the tests' fake model, as the fold's test fixtures.

Each scenario runs deep_reasoner in a worker, as the app does, with tests/fake_model.py
answering on 127.0.0.1; its events, as the app file holds them, are written to
tests/runs/<scenario>.jsonl, one JSON line each, with this machine's site-packages folder,
which tracebacks name, written as SITE, and the runs' temporary home as HOME.

Run from the repo root: uv run python scripts/record_runs.py [scenario ...], every
scenario unless some are named.
"""

import asyncio
import json
import os
import sys
import sysconfig
import tempfile
from contextlib import closing
from pathlib import Path

import structlog

TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS))

from fake_model import (
    EXAMPLE,
    RESEARCH,
    RESEARCH_CHANGED,
    REWORDED,
    SPLIT_COUNT,
    WORD_COUNT,
    FakeOpenAI,
    library_cell,
    repl,
    scripted,
    task_and_turn,
    write,
)

from deep_reasoning.library import (
    ROOT,
    add_namespace,
    engine_config,
    has_namespace,
    open_library,
)
from deep_reasoning.runs import Runs, read_events

OUT = TESTS / "runs"
SITE_PACKAGES = sysconfig.get_paths()["purelib"]
CHAT = "recorded"
LIBRARY = "library.sqlite"
PRICED = "gpt-6-luna"
UNPRICED = "a-model-nobody-prices"
SPIN = "while True:\n    pass"
SLEEP = "import time\ntime.sleep(60)"
# A cell blocked in C holds up a clean stop, so the server kills the worker this soon.
KILL_AFTER_S = 0.5
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
SCENARIOS = {
    "fanout": {"prompt": "Summarize two courses.", "plan": FAN_OUT},
    "unpriced": {
        "prompt": "Summarize two courses.",
        "plan": FAN_OUT,
        "settings": {"model": UNPRICED},
    },
    "nested": {
        "prompt": "Plan the term.",
        "plan": {
            "Plan the term.": [
                repl("plan = subagent('Survey CS.')", "print(plan)"),
                repl("FinalAnswer(plan)"),
            ],
            "Survey CS.": [
                repl("course = subagent('Read CS101.')", "print(course)"),
                repl("FinalAnswer('CS: ' + course)"),
            ],
            "Read CS101.": [repl("FinalAnswer('CS101 is light')")],
        },
    },
    "drives": {
        "prompt": "Remember and fork.",
        "plan": {
            "Remember and fork.": [
                repl(
                    "s = subagent()",
                    "first = run_all([anext(s.send('Note that the answer is 42.'))])",
                    "again = run_all([anext(s.send('Say it again.'))])",
                    "print(first, again)",
                ),
                repl("FinalAnswer([first, again])"),
            ],
            "Note that the answer is 42.": [repl("FinalAnswer(42)")],
            "Say it again.": [repl("FinalAnswer('again 42')")],
        },
    },
    "failing": {
        "prompt": "Survive failures.",
        "plan": {
            "Survive failures.": [
                repl("print(1 / 0)", think="This cell fails."),
                repl("r = subagent('Break the model.')", "print(r)"),
                repl("FinalAnswer('recovered')"),
            ],
            "Break the model.": [repl("print('trying')")],
        },
        "failing": ("Break the model.", 1),
    },
    "raising": {
        "prompt": "Raise three ways.",
        "plan": {
            "Raise three ways.": [
                repl("print(subagent('Raise them.'))"),
                repl("FinalAnswer('raised')"),
            ],
            "Raise them.": [
                repl("print('counting')"),
                repl("raise ValueError('2 problems:\\ntools: bad\\nspawn: bad')"),
                repl("raise ExceptionGroup('x', [ValueError(1), KeyError(2)])"),
                repl("FinalAnswer('x')", "raise ValueError('after')"),
            ],
        },
    },
    "exhausted": {
        "prompt": "Never finish.",
        "plan": {
            "Never finish.": [
                repl(
                    "r = run_all([anext(subagent().send('Loop forever.'))])", "print(r)"
                ),
                repl("print('still thinking')"),
            ],
            "Loop forever.": [repl("print('looping')")],
        },
        "settings": {"max_iter": 3},
    },
    "stopped": {
        "prompt": "Keep a spinner going.",
        "plan": {
            "Keep a spinner going.": [repl("print(subagent('Spin.'))")],
            "Spin.": [repl("print('warming up')"), repl(*SPIN.split("\n"))],
        },
        "stop_at": SPIN,
    },
    "killed": {
        "prompt": "Wait a minute.",
        "plan": {
            "Wait a minute.": [repl("print(subagent('Sleep.'))")],
            "Sleep.": [repl(*SLEEP.split("\n"))],
        },
        "stop_at": SLEEP,
    },
    "librarian": {
        "prompt": "File the example.",
        "namespace": "librarian",
        "plan": {
            "File the example.": [
                write("librarian", EXAMPLE),
                repl("print(subagent('Reword the example.', namespace='librarian'))"),
                repl("FinalAnswer('filed')"),
            ],
            "Reword the example.": [
                write("librarian", REWORDED),
                repl("FinalAnswer('reworded')"),
            ],
        },
    },
    "namespaces": {
        "prompt": "Set up research.",
        "namespace": "librarian",
        "plan": {
            "Set up research.": [
                library_cell("add_namespace", "research", RESEARCH),
                library_cell("change_namespace", "research", RESEARCH_CHANGED),
                repl("FinalAnswer('set up')"),
            ],
        },
    },
    "pages": {
        "prompt": "Show your work.",
        "namespace": "shown",
        "namespaces": {"shown": "tools: [page]\n"},
        "plan": {
            "Show your work.": [
                repl(
                    "p = page.new('How I counted')",
                    "p.text('By **hand**.')",
                    "print(p.open())",
                ),
                repl("print(subagent('Table the seats.', namespace='shown'))"),
                repl("FinalAnswer('shown')"),
            ],
            "Table the seats.": [
                repl(
                    "p = page.new('Seats')",
                    "p.table([{'course': 'CS101', 'seats': 3}])",
                    "print(p.open())",
                ),
                repl("FinalAnswer('tabled')"),
            ],
        },
    },
    "tools": {
        "prompt": "Make a word counter.",
        "namespace": "librarian",
        "plan": {
            "Make a word counter.": [
                library_cell("write_tool", "word_count", WORD_COUNT, "build"),
                library_cell("add_namespace", "counting", "tools: [word_count]\n"),
                library_cell(
                    "write_tool", "word_count", SPLIT_COUNT, "build", {"sep": ","}
                ),
                repl("FinalAnswer('made')"),
            ],
        },
    },
}

log = structlog.get_logger()


def responder(scenario):
    """The scenario's plan, answering HTTP 500 at its failing (task, turn), if any."""
    follow = scripted(scenario["plan"])

    def respond(messages):
        task, turn = task_and_turn(messages)
        failing = scenario.get("failing")
        if failing and failing[0] in task and failing[1] == turn:
            raise RuntimeError("the fake model is down")
        return follow(messages)

    return respond


def config_on(fake, settings):
    """The engine config of a run whose model is *fake*."""
    client = {"base_url": fake.base_url, "api_key_env": "OPENAI_API_KEY"}
    return {
        "model": PRICED,
        "client": client | {"max_retries": 0},
        "system_prompt": "You are a scripted test agent.",
        "max_iter": 6,
    } | settings


async def read_to_the_end(runs, run, stop_at):
    """Follow the run until it ends, stopping it once a cell running *stop_at* starts."""
    async for event in runs.events(run):
        if event["kind"] == "cell.start" and event["code"] == stop_at:
            runs.stop(run)


async def record(home, name, scenario):
    library = home / LIBRARY
    namespace = scenario.get("namespace", ROOT)
    with closing(open_library(library)) as connection:
        for added, text in scenario.get("namespaces", {}).items():
            if not has_namespace(connection, added):
                add_namespace(connection, added, text)
    with FakeOpenAI(responder(scenario)) as fake:

        def config_for(folder):
            # The Library's config, as the app builds a run's, with the fake's model.
            told = engine_config(
                library, namespace, folder / "tools", CHAT, folder.name
            )
            return told | config_on(fake, scenario.get("settings", {}))

        runs = Runs(library, home, stop_grace_s=KILL_AFTER_S)
        run = runs.start(CHAT, scenario["prompt"], config_for, [])
        await read_to_the_end(runs, run, scenario.get("stop_at"))
    with closing(open_library(library)) as connection:
        events = read_events(connection, run)
    lines = "".join(json.dumps(event) + "\n" for event in events)
    lines = lines.replace(SITE_PACKAGES, "SITE").replace(str(home), "HOME")
    (OUT / f"{name}.jsonl").write_text(lines)
    log.info("Recorded", scenario=name, run=run)


async def main():
    os.environ["OPENAI_API_KEY"] = "fake"
    OUT.mkdir(exist_ok=True)
    names = sys.argv[1:] or SCENARIOS
    with tempfile.TemporaryDirectory() as home:
        for name in names:
            await record(Path(home), name, SCENARIOS[name])


if __name__ == "__main__":
    asyncio.run(main())
