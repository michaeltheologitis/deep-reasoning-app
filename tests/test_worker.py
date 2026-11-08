import asyncio
import io
import json

import pytest
from deep_reasoner.tools import ToolBuildError
from deep_reasoner.v2.cli import V2Config
from deep_reasoner.v2.messages import code
from deep_reasoner.v2.repl_coro import REPL
from genai_prices import Usage, calc_price

from deep_reasoning import worker
from deep_reasoning.worker import (
    Recorder,
    build_failure,
    cost_of,
    final_answer_of,
    output_of,
)

ROOT, CHILD = 1, 2
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


def recorded(*engine_events):
    """The app's events a Recorder writes for deep_reasoner's *engine_events*."""
    out = io.StringIO()
    recorder = Recorder(out, on_root_start=lambda: None)
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
    assert events[-1]["node"] == ROOT


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


def test_nothing_is_written_after_the_turn_ends():
    out = io.StringIO()
    recorder = Recorder(out, on_root_start=lambda: None)
    recorder(None, "debug", agent_start(ROOT, None))
    recorder.emit("turn.end", outcome="answered", answer="done", detail=None)
    recorder(None, "debug", agent_start(CHILD, ROOT))
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["kind"] for line in lines] == ["agent.start", "turn.end"]


def test_the_roots_start_is_reported():
    started = []
    recorder = Recorder(io.StringIO(), on_root_start=lambda: started.append(True))
    recorder(None, "debug", agent_start(ROOT, None))
    recorder(None, "debug", agent_start(CHILD, ROOT))
    assert started == [True]


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
    expected = float(calc_price(tokens, "gpt-6-luna").total_price)
    assert cost_of("gpt-6-luna", usage) == pytest.approx(expected)
    assert cost_of("gpt-6-luna", usage) < cost_of(
        "gpt-6-luna", usage | {"prompt_tokens_details": {}}
    )


def test_the_providers_own_cost_wins():
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.5}
    assert cost_of("gpt-6-luna", usage) == 0.5


def test_an_unknown_models_cost_is_none_not_zero():
    assert (
        cost_of("no-such-model", {"prompt_tokens": 10, "completion_tokens": 2}) is None
    )


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
