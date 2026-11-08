import asyncio
import io
import json
import signal
import subprocess
import sys

import pytest
from deep_reasoner.tools import ToolBuildError
from deep_reasoner.v2 import agent
from deep_reasoner.v2.cli import V2Config, build_reasoner, close_run
from deep_reasoner.v2.messages import code, func
from deep_reasoner.v2.repl_coro import REPL
from genai_prices import Usage, calc_price

from deep_reasoning import worker
from deep_reasoning.runs import LEFT_OUT
from deep_reasoning.worker import (
    Recorder,
    build_failure,
    cost_of,
    final_answer_of,
    output_of,
    tool_line,
)

ROOT, CHILD = 1, 2
# The engine's own, as it is before a run names any tool: tool_line words lines as it does.
ENGINE_DESCRIBE = func.describe
OPENAI_URL = "https://api.openai.com/v1"
OWN_SERVER_URL = "http://127.0.0.1:8000/v1"
EXIT_S = 30
# Non-blocking: a read that would wait for a command raises, ending the group, so the
# test fails rather than hangs.
TAKE_A_STOP_TWICE = """\
import os
from deep_reasoning.worker import take_commands
read, write = os.pipe()
os.write(write, b"stop\\n")
os.set_blocking(read, False)
print([take_commands(read, lambda: print("stopped")) for _ in range(2)])
"""
READ_COMMANDS_FROM_A_FOLDER = """\
import os, sys
from deep_reasoning.worker import read_commands
read_commands(os.open(sys.argv[1], os.O_RDONLY), lambda: print("stopped"))
"""
TASK_GROUP = """\
import asyncio, concurrent.futures
async def boom():
    raise ValueError('room D')
async def main():
    async with asyncio.TaskGroup() as group:
        group.create_task(boom())
concurrent.futures.ThreadPoolExecutor().submit(asyncio.run, main()).result()"""
RAISING = {
    "alone": "{}[3]",
    "after output": "print('counting')\n{}[3]",
    "message over lines": "raise ValueError('2 problems:\\ntools: bad\\nspawn: bad')",
    "chained": "try:\n    {}[3]\nexcept KeyError:\n    raise ValueError('no key')",
    "a syntax error": "print(",
    "after its answer": "FinalAnswer('x')\nraise ValueError('after')",
    "an exception group": "raise ExceptionGroup('x', [ValueError(1), KeyError(2)])",
    "a task group": TASK_GROUP,
}


def new_recorder(out, on_root_start=lambda: None):
    """A Recorder writing to *out*, of a run whose model calls go to OpenAI."""
    return Recorder(out, on_root_start=on_root_start, base_url=OPENAI_URL)


def recorded(*engine_events):
    """The app's events a Recorder writes for deep_reasoner's *engine_events*."""
    out = io.StringIO()
    recorder = new_recorder(out)
    for event in engine_events:
        recorder(None, "debug", dict(event))
    return [json.loads(line) for line in out.getvalue().splitlines()]


def agent_start(node, parent, task="a task"):
    return {
        "event": "agent.start",
        "node_id": node,
        "agent_id": node,
        "parent": parent,
        "task": task,
        "namespace": "root",
        "messages": [{"role": "system", "content": "the whole prompt"}],
    }


def model_reply(node, response):
    return {
        "event": "llm.call",
        "node_id": node,
        "agent_id": node,
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "the conversation so far"}],
        "response": response,
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def cell_ran(node, source, printed):
    return {
        "event": "repl.execute",
        "node_id": node,
        "agent_id": node,
        "source": source,
        "observation": f"<observation>\n{printed}\n</observation>",
    }


def observed(source):
    """What deep_reasoner's REPL observes of running *source*, without the tags."""
    repl = REPL()
    repl.send(code(f"<repl>\n{source}\n</repl>"))
    return output_of(asyncio.run(anext(repl)).text)


def agent_end(node, status):
    return {"event": "agent.end", "node_id": node, "agent_id": node, "status": status}


def test_a_reply_becomes_a_thought_and_the_cell_it_asks_for():
    events = recorded(
        agent_start(ROOT, None),
        model_reply(ROOT, "<think>Count them.</think>\n<repl>\nprint(3)\n</repl>"),
        cell_ran(ROOT, "print(3)", "3"),
    )
    steps = [event for event in events if event["kind"] in ("think", "cell.start")]
    assert steps == [
        {"kind": "think", "node": ROOT, "text": "Count them."},
        {"kind": "cell.start", "node": ROOT, "cell": 1, "code": "print(3)"},
    ]
    assert events[-1] == {
        "kind": "cell.end",
        "node": ROOT,
        "cell": 1,
        "code": "print(3)",
        "output": "3",
        "status": "done",
    }


def test_a_sub_agent_started_outside_a_cell_gets_an_empty_cell_of_its_parent():
    events = recorded(agent_start(ROOT, None), agent_start(CHILD, ROOT))
    assert events[1] == {"kind": "cell.start", "node": ROOT, "cell": 1, "code": ""}
    assert events[2]["kind"] == "agent.start" and events[2]["cell"] == 1


def test_a_cell_open_when_its_agent_stops_ends_stopped():
    events = recorded(
        agent_start(ROOT, None),
        model_reply(ROOT, "<repl>\nwhile True:\n    pass\n</repl>"),
        agent_end(ROOT, "stopped"),
    )
    assert events[-2] == {
        "kind": "cell.end",
        "node": ROOT,
        "cell": 1,
        "code": "while True:\n    pass",
        "output": "",
        "status": "stopped",
    }
    assert events[-1]["kind"] == "agent.end" and events[-1]["status"] == "stopped"


@pytest.mark.parametrize("source", RAISING.values(), ids=RAISING.keys())
def test_a_cell_that_raised_failed(source):
    events = recorded(agent_start(ROOT, None), cell_ran(ROOT, source, observed(source)))
    assert events[-1]["status"] == "failed"


def test_a_cell_that_prints_a_recorded_traceback_inside_data_is_done():
    steps = [{"kind": "cell", "output": observed("{}[3]"), "status": "failed"}]
    source = f"print({steps!r})"
    events = recorded(agent_start(ROOT, None), cell_ran(ROOT, source, observed(source)))
    assert events[-1]["status"] == "done"


def test_a_cell_that_prints_a_traceback_it_caught_and_carries_on_is_done():
    source = """\
import traceback
try:
    {}[3]
except KeyError:
    print(traceback.format_exc())
print('carried on')"""
    events = recorded(agent_start(ROOT, None), cell_ran(ROOT, source, observed(source)))
    assert events[-1]["status"] == "done"


def test_an_agent_that_answers_has_its_final_answer_as_result():
    events = recorded(
        agent_start(ROOT, None),
        cell_ran(ROOT, "FinalAnswer('C1: 3')", "FinalAnswer: 'C1: 3'"),
        agent_end(ROOT, "done"),
    )
    assert events[-1] == {
        "kind": "agent.end",
        "node": ROOT,
        "status": "done",
        "result": "C1: 3",
        "detail": None,
    }


def test_the_conversation_is_never_forwarded():
    events = recorded(
        agent_start(ROOT, None), model_reply(ROOT, "<repl>\nprint(1)\n</repl>")
    )
    assert all("messages" not in event for event in events)
    assert all("the whole prompt" not in json.dumps(event) for event in events)


def test_a_tools_model_call_is_priced_for_its_agent_but_is_not_a_thought():
    call = model_reply(3, "a summary") | {"agent_id": ROOT}
    events = recorded(agent_start(ROOT, None), call)
    assert [event["kind"] for event in events] == ["agent.start", "usage"]
    assert (events[-1]["node"], events[-1]["own"]) == (ROOT, False)
    assert events[-1]["cost"] > 0


def test_an_agents_own_call_says_so_with_its_model():
    events = recorded(agent_start(ROOT, None), model_reply(ROOT, "Thinking."))
    usage = next(event for event in events if event["kind"] == "usage")
    assert (usage["own"], usage["model"]) == (True, "gpt-6-luna")


def test_a_library_change_is_recorded_for_the_agent_whose_cell_made_it():
    change = {
        "event": "library.change",
        "node_id": CHILD,
        "agent_id": CHILD,
        "what": "decomposition",
        "namespace": "librarian",
        "name": "count",
        "before": None,
        "after": "name: count\n",
    }
    events = recorded(
        agent_start(ROOT, None),
        agent_start(CHILD, ROOT),
        model_reply(
            CHILD, "<repl>\nlibrary.write_decomposition('librarian', y)\n</repl>"
        ),
        change,
    )
    [cell] = [event["cell"] for event in events if event["kind"] == "cell.start"][-1:]
    assert events[-1] == {
        "kind": "library.change",
        "node": CHILD,
        "cell": cell,
        "what": "decomposition",
        "namespace": "librarian",
        "name": "count",
        "before": None,
        "after": "name: count\n",
    }


def test_a_page_opened_is_recorded_for_the_agent_whose_cell_opened_it():
    opened = {
        "event": "page.open",
        "node_id": CHILD,
        "agent_id": CHILD,
        "page": "eager-modest-oarfish",
        "title": "How I counted",
    }
    events = recorded(
        agent_start(ROOT, None),
        agent_start(CHILD, ROOT),
        model_reply(CHILD, "<repl>\nprint(page.new('How I counted').open())\n</repl>"),
        opened,
    )
    [cell] = [event["cell"] for event in events if event["kind"] == "cell.start"][-1:]
    assert events[-1] == {
        "kind": "page.open",
        "node": CHILD,
        "cell": cell,
        "page": "eager-modest-oarfish",
        "title": "How I counted",
    }


def test_the_turns_end_counts_the_chat_calls_started_that_never_finished():
    out = io.StringIO()
    recorder = new_recorder(out)
    recorder(None, "debug", agent_start(ROOT, None))
    for _ in range(3):
        recorder.call_started()
    recorder(None, "debug", model_reply(ROOT, "Thinking."))
    recorder(None, "debug", model_reply(ROOT, "Still thinking."))
    recorder.end_turn("stopped", None, None)
    end = json.loads(out.getvalue().splitlines()[-1])
    assert (end["kind"], end["unfinished_calls"]) == ("turn.end", 1)


def test_nothing_is_written_after_the_turns_end():
    out = io.StringIO()
    recorder = new_recorder(out)
    recorder(None, "debug", agent_start(ROOT, None))
    recorder.end_turn("answered", "done", None)
    recorder(None, "debug", agent_start(CHILD, ROOT))
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["kind"] for line in lines] == ["agent.start", "turn.end"]


def test_the_roots_start_is_reported():
    started = []
    recorder = new_recorder(io.StringIO(), on_root_start=lambda: started.append(True))
    recorder(None, "debug", agent_start(ROOT, None))
    recorder(None, "debug", agent_start(CHILD, ROOT))
    assert started == [True]


DIVIDING_BY_ZERO = {
    "name": "tally",
    "messages": [
        {"role": "user", "content": "Tally {{ 1/0 }} votes."},
        {
            "role": "assistant",
            "content": "<think>t</think>\n<repl>FinalAnswer(1)</repl>",
        },
    ],
}


def unstarted(folder):
    """A run's root, not started, whose namespace holds a worked example no run renders."""
    config = V2Config.model_validate(
        {
            "model": "gpt-6-luna",
            "client": {"base_url": OWN_SERVER_URL, "api_key_env": "OPENAI_API_KEY"},
            "namespaces": {"root": {"decompositions": [DIVIDING_BY_ZERO]}},
        }
    )
    reasoner, _ = build_reasoner(config, run_dir=folder)
    return reasoner.send("Tally the votes.")


def test_a_worked_example_left_out_is_noted_for_the_agent_it_was_left_out_of_alone(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    # put back after the test, as the wrapping replaces them
    monkeypatch.setattr(agent, "_rendered_turns", agent._rendered_turns)
    base = agent.DeepReasonerAgentBase
    monkeypatch.setattr(base, "_start", base._start)
    out = io.StringIO()
    recorder = new_recorder(out)
    worker.leaving_out_unrendered(recorder)
    failing, started = unstarted(tmp_path / "failing"), unstarted(tmp_path / "started")

    def no_model():
        raise RuntimeError("no model")

    # It fails after rendering its examples, so it never says it started.
    monkeypatch.setattr(failing, "_build_neural", no_model)
    with pytest.raises(RuntimeError, match="no model"):
        failing._start()
    recorder(None, "debug", agent_start(CHILD, None))
    started._start()
    node = started._log_vars["node_id"]
    recorder(None, "debug", agent_start(node, None))
    close_run(failing)
    close_run(started)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [(line["kind"], line["node"]) for line in lines] == [
        ("agent.start", CHILD),
        ("agent.start", node),
        (LEFT_OUT, node),
    ]
    assert lines[-1]["why"] == "ZeroDivisionError: division by zero"


def in_a_worker(code, *args):
    """*code* run in a process group of its own, as the worker is: ending its group
    ends it alone."""
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        check=False,
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=EXIT_S,
    )


def test_a_stop_already_sent_is_taken_once_without_waiting_for_more():
    taken = in_a_worker(TAKE_A_STOP_TWICE)
    assert (taken.returncode, taken.stdout) == (0, "stopped\n[True, False]\n")


def test_a_stdin_that_cant_be_read_ends_the_worker_as_its_end_does(tmp_path):
    reading = in_a_worker(READ_COMMANDS_FROM_A_FOLDER, str(tmp_path))
    assert reading.returncode == -signal.SIGKILL, reading.stderr
    assert "stopped" not in reading.stdout


@pytest.mark.parametrize(
    ("output", "answer"),
    [
        ("FinalAnswer: 'C1: 3'", "'C1: 3'"),
        ("thinking aloud\nFinalAnswer: {'a': 1}", "{'a': 1}"),
        ("no answer yet", None),
        ("FinalAnswer: 'printed'\nFinalAnswer: 'returned'", "'returned'"),
    ],
)
def test_the_final_answer_is_read_from_a_cells_output(output, answer):
    assert final_answer_of(output) == answer


def test_cached_input_is_billed_at_its_own_rate():
    usage = {
        "prompt_tokens": 10_000,
        "completion_tokens": 2_000,
        "prompt_tokens_details": {"cached_tokens": 8_000},
    }
    tokens = Usage(input_tokens=10_000, cache_read_tokens=8_000, output_tokens=2_000)
    expected = float(calc_price(tokens, "gpt-6-luna", provider_id="openai").total_price)
    assert cost_of("gpt-6-luna", usage, OPENAI_URL) == pytest.approx(expected)
    assert cost_of("gpt-6-luna", usage, OPENAI_URL) < cost_of(
        "gpt-6-luna", usage | {"prompt_tokens_details": {}}, OPENAI_URL
    )


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("gpt-6-luna", "https://api.openai.com/v1"),
        ("openai/gpt-6-luna", "https://openrouter.ai/api/v1"),
        ("claude-sonnet-5-5", "https://api.anthropic.com/v1/"),
        (
            "gemini-3.8-flash",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    ],
)
def test_a_call_to_a_vendor_is_priced_from_that_vendors_prices(model, base_url):
    assert cost_of(model, {"prompt_tokens": 10, "completion_tokens": 2}, base_url) > 0


def test_a_vendors_model_on_a_server_of_your_own_is_unpriced():
    usage = {"prompt_tokens": 10, "completion_tokens": 2}
    assert cost_of("gpt-6-luna", usage, OWN_SERVER_URL) is None


def test_the_providers_own_cost_wins():
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.5}
    assert cost_of("gpt-6-luna", usage, OWN_SERVER_URL) == 0.5
    assert cost_of("gpt-6-luna", usage, OPENAI_URL) == 0.5


def test_an_unknown_models_cost_is_none_not_zero():
    usage = {"prompt_tokens": 10, "completion_tokens": 2}
    assert cost_of("no-such-model", usage, OPENAI_URL) is None


TOOLS = "/data/runs/20261108-000000-000000/tools"
BROKEN = {"factory": "build", "factory_from": f"{TOOLS}/broken.py"}
BUILD_ERROR = ToolBuildError(
    "broken", "factory", f"build in {TOOLS}/broken.py raised RuntimeError: no index"
)


@pytest.mark.parametrize(
    ("namespaces", "where"),
    [
        (
            {"units": {"tools": ["broken"]}, "units.metric": {"tools": ["broken"]}},
            "It is granted in units, units.metric: take it out of their tools",
        ),
        ({}, "Take it out of the tools of each namespace that grants it"),
    ],
    ids=["granted", "no-grant-found"],
)
def test_a_run_whose_tool_doesnt_build_says_which_where_and_how_to_take_it_out(
    namespaces, where
):
    config = V2Config.model_validate(
        {"tools": {"broken": BROKEN}, "namespaces": namespaces}
    )
    assert build_failure(BUILD_ERROR, config) == (
        "Tool broken doesn't build, so no chat can run while a namespace grants it: build"
        f" in broken.py raised RuntimeError: no index\n{where} in the Library panel's"
        " Namespaces tab (Edit YAML)."
    )


def test_a_run_whose_namespaces_cant_be_read_still_says_which_tool_doesnt_build(
    monkeypatch,
):
    def unreadable(config):
        raise ValueError("no registry")

    monkeypatch.setattr(worker, "build_namespace_registry", unreadable)
    config = V2Config.model_validate({"tools": {"broken": BROKEN}})
    assert build_failure(BUILD_ERROR, config).endswith(
        "no index\nTake it out of the tools of each namespace that grants it in the"
        " Library panel's Namespaces tab (Edit YAML)."
    )


def count_words(text: str, sep: str | None = None) -> int:
    """How many words text has."""
    return len(text.split(sep))


def undocumented(x, *rest, scale=1, **options):
    return x


@pytest.mark.parametrize(
    ("value", "description"),
    [(count_words, ""), (count_words, "Counts words."), (undocumented, ""), (len, "")],
    ids=["docstring", "description", "nothing", "a-builtin"],
)
def test_a_tool_named_as_its_value_is_shown_as_the_engine_shows_it(value, description):
    bound = func(value.__name__, value, description)
    shown = tool_line(bound.name, bound.value, bound.description)
    assert shown == ENGINE_DESCRIBE(bound)
