import json
import re
import sqlite3
from contextlib import closing
from itertools import count
from pathlib import Path

import pytest
import yaml
from fake_model import RESEARCH_CHANGED, SPLIT_COUNT, WORD_COUNT
from structlog.testing import capture_logs

from deep_reasoning.app_tools import (
    BRIEF_CHARS,
    CHANGED_MEANWHILE,
    CUT_SHORT,
    EMPTY,
    MAX_FOUND,
    MAX_VALUE_CHARS,
    MAX_VALUE_DEPTH,
    MAX_YAML_CHARS,
    NESTED_TOO_DEEP,
    NOT_A_REF,
    NOT_A_TOOL,
    NOT_AN_INDEX,
    NOT_FINDABLE,
    NOT_LOCAL,
    NOT_MARKDOWN,
    NOT_ROWS,
    NOT_SAVED,
    NOT_SHOWN,
    NOT_TEXT,
    NOT_VALUES,
    SETS_OWN_KEYS,
    TITLE,
    TOO_DEEP,
    build_history,
    build_library,
    build_page,
    open_existing,
)
from deep_reasoning.library import (
    HISTORIAN,
    HISTORIANS_EXAMPLES,
    HISTORY_DESCRIPTION,
    LIBRARIAN,
    LIBRARIAN_HAND_OFFS,
    LIBRARIANS_EXAMPLES,
    LIBRARY_DESCRIPTION,
    UPSERT_TOOL,
    LibraryError,
    add_namespace,
    change_namespace,
    delete_namespace,
    dump,
    open_library,
    parsed,
    writing,
)
from deep_reasoning.pages import MAX_PAGE_CHARS, read_page
from deep_reasoning.runs import (
    CHANGE_FIELDS,
    KEY_WITHHELD,
    LOST,
    PAGE_FIELDS,
    SPEC,
    chat_runs,
    read_turn,
    save_event,
)

RUNS = Path(__file__).parent / "runs"
CHAT = "a-chat"
ANOTHER_CHAT = "another-chat"
CURRENT = "20261027-235959-000000"
RUN_NUMBERS = count()
EXHAUSTED = "Agent failed to produce a final answer within 3 iterations."


def recorded(name):
    """A recorded run's events, as the app file holds them."""
    lines = (RUNS / f"{name}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def plant(library, chat, events):
    """*events* as a run of *chat* in the app file; its id."""
    run = f"20261027-120000-{next(RUN_NUMBERS):06d}"
    with closing(open_library(library)) as connection:
        for event in events:
            if event["kind"] == "turn.start":
                event = event | {"run": run, "chat": chat}
            save_event(connection, run, event)
    return run


def history_of(library, chat=CHAT, run=CURRENT):
    """The history tool, as the run *run* of *chat* gets it."""
    told = {"library": str(library), "chat": chat, "run": run}
    return build_history(None, told).value


def cost_of_calls(events, nodes):
    return sum(
        event["cost"]
        for event in events
        if event["kind"] == "usage" and event["node"] in nodes
    )


def test_turns_number_the_chats_earlier_messages_as_a_follow_up_gets_them(library):
    plant(library, CHAT, recorded("nested"))
    plant(library, CHAT, [{"seq": 1, "t": 1.0, "kind": "usage"}])
    plant(library, CHAT, recorded("failing"))
    current = plant(library, CHAT, recorded("fanout"))
    turns = history_of(library, run=current).turns()
    with closing(open_library(library)) as connection:
        earlier = [read_turn(connection, run) for run in chat_runs(connection, CHAT)]
    expected = [turn.message for turn in earlier if turn is not None][:-1]
    assert [turn["message"] for turn in turns] == expected
    assert [(turn["turn"], turn["message"], turn["answer"]) for turn in turns] == [
        (1, "Plan the term.", "CS: CS101 is light"),
        (2, "Survive failures.", "recovered"),
    ]
    assert turns[0]["cost"] == pytest.approx(
        cost_of_calls(recorded("nested"), {1, 2, 3})
    )
    assert [turn["agents"] for turn in turns] == [2, 1]


def test_another_chats_runs_are_never_seen(library):
    plant(library, CHAT, recorded("nested"))
    plant(library, ANOTHER_CHAT, recorded("drives"))
    assert [turn["message"] for turn in history_of(library).turns()] == [
        "Plan the term."
    ]
    assert [turn["message"] for turn in history_of(library, ANOTHER_CHAT).turns()] == [
        "Remember and fork."
    ]


def test_an_agents_cost_counts_its_own_calls_and_its_sub_agents(library):
    plant(library, CHAT, recorded("nested"))
    survey, read = history_of(library).agents(1)
    events = recorded("nested")
    assert [(agent["node"], agent["parent"]) for agent in (survey, read)] == [
        (2, 1),
        (3, 2),
    ]
    assert survey["cost"] == pytest.approx(cost_of_calls(events, {2, 3}))
    assert read["cost"] == pytest.approx(cost_of_calls(events, {3}))
    assert (survey["result"], survey["steps"], read["steps"]) == (
        "CS: CS101 is light",
        2,
        1,
    )


def test_an_agent_is_timed_from_its_start_to_its_end_to_the_tenth_as_the_tree_shows_it(
    library,
):
    plant(library, CHAT, recorded("nested"))
    events = recorded("nested")
    times = {
        (event["kind"], event["node"]): event["t"]
        for event in events
        if event["kind"] in ("agent.start", "agent.end")
    }
    survey, read = history_of(library).agents(1)
    for agent in (survey, read):
        node = agent["node"]
        took = times["agent.end", node] - times["agent.start", node]
        assert agent["seconds"] == round(took, 1)


def test_steps_drill_down_through_the_sub_agents_each_cell_started(library):
    plant(library, CHAT, recorded("nested"))
    history = history_of(library)
    assert [step["agents"] for step in history.steps(1)] == [[2], []]
    [delegated, answered] = history.steps(1, node=2)
    assert delegated["code"] == "course = subagent('Read CS101.')\nprint(course)"
    assert (delegated["output"], delegated["agents"]) == ("CS101 is light", [3])
    assert answered["output"] == "FinalAnswer: 'CS: CS101 is light'"
    assert history.steps(1, node=3)[0]["code"] == "FinalAnswer('CS101 is light')"


def test_a_fan_out_lists_each_sub_agent_under_the_cell_that_started_them(library):
    plant(library, CHAT, recorded("fanout"))
    history = history_of(library)
    thought, fanned, _ = history.steps(1)
    assert (thought["kind"], thought["text"]) == ("think", "Fan out.")
    assert fanned["agents"] == [2, 3]
    assert [agent["task"] for agent in history.agents(1)] == [
        "Summarize C1.",
        "Summarize C2.",
    ]


def test_a_sub_agent_driven_twice_has_a_row_per_drive_and_its_steps_in_order(library):
    plant(library, CHAT, recorded("drives"))
    history = history_of(library)
    drives = [(agent["node"], agent["drive"]) for agent in history.agents(1)]
    assert drives == [(2, 1), (2, 2)]
    assert history.steps(1)[0]["agents"] == [2]
    assert [step["drive"] for step in history.steps(1, node=2)] == [1, 2]


def as_change_step(event):
    """A library.change event as history.steps lists it, in a first drive."""
    return (
        {"kind": "change"} | {key: event[key] for key in CHANGE_FIELDS} | {"drive": 1}
    )


def test_steps_list_each_library_change_with_the_agent_that_made_it(library):
    events = recorded("librarian")
    plant(library, CHAT, events)
    history = history_of(library)
    added, replaced = [event for event in events if event["kind"] == "library.change"]
    [reworder] = history.agents(1)
    top = history.steps(1)
    assert [step["kind"] for step in top] == ["cell", "change", "cell", "cell"]
    assert top[1] == as_change_step(added)
    assert history.steps(1, node=reworder["node"])[1] == as_change_step(replaced)
    assert (added["before"], replaced["before"]) == (None, added["after"])


def test_steps_list_each_page_opened_with_the_agent_that_opened_it(library):
    events = recorded("pages")
    plant(library, CHAT, events)
    history = history_of(library)
    top, tabled = [event for event in events if event["kind"] == "page.open"]
    [tabler] = history.agents(1)
    as_step = {"kind": "page"} | {key: top[key] for key in PAGE_FIELDS} | {"drive": 1}
    assert [step["kind"] for step in history.steps(1)] == [
        "cell",
        "page",
        "cell",
        "cell",
    ]
    assert history.steps(1)[1] == as_step
    assert history.steps(1, node=tabler["node"])[1]["title"] == tabled["title"]


@pytest.mark.parametrize("name", ["stopped", "killed"])
def test_what_a_stopped_run_left_going_reads_as_stopped(library, name):
    plant(library, CHAT, recorded(name))
    history = history_of(library)
    [agent] = history.agents(1)
    going = [history.steps(1)[-1], history.steps(1, node=agent["node"])[-1]]
    assert agent["status"] == "stopped"
    assert [cell["status"] for cell in going] == ["stopped", "stopped"]


def test_a_run_the_app_stopped_under_reads_as_failed(library):
    plant(library, CHAT, recorded("killed")[:-1])
    history = history_of(library)
    [turn] = history.turns()
    assert (turn["outcome"], turn["detail"]) == ("failed", LOST)
    assert history.agents(1)[0]["status"] == "stopped"


def test_an_exhausted_run_and_its_exhausted_sub_agent_read_as_such(library):
    plant(library, CHAT, recorded("exhausted"))
    history = history_of(library)
    [turn] = history.turns()
    [looping] = history.agents(1)
    assert (turn["outcome"], turn["answer"]) == ("exhausted", EXHAUSTED)
    assert (looping["status"], looping["result"], looping["steps"]) == (
        "exhausted",
        None,
        3,
    )
    assert looping["cost"] == pytest.approx(cost_of_calls(recorded("exhausted"), {2}))


def crafted(*events):
    """A run's rows as a file from someone else may hold them: a turn.start, *events*,
    then a turn.end."""
    rows = [{"kind": "turn.start", "prompt": "Crafted."}, *events]
    rows.append({"kind": "turn.end", "outcome": "answered", "answer": "crafted"})
    return [{"seq": seq, "t": float(seq)} | row for seq, row in enumerate(rows, 1)]


ROOT_AND_CELL = (
    {"kind": "agent.start", "node": 1, "parent": None, "cell": None},
    {"kind": "cell.start", "node": 1, "cell": 1, "code": "print('kept')"},
)


def test_a_sub_agent_started_again_as_the_root_leaves_history_answering(library):
    plant(library, CHAT, recorded("nested"))
    plant(
        library,
        CHAT,
        crafted(
            *ROOT_AND_CELL,
            {"kind": "agent.start", "node": 2, "parent": 1, "cell": 1},
            {"kind": "agent.start", "node": 2, "parent": None, "cell": None},
            {"kind": "usage", "node": 2, "cost": 0.5},
        ),
    )
    history = history_of(library)
    assert [turn["message"] for turn in history.turns()] == [
        "Plan the term.",
        "Crafted.",
    ]
    assert [(agent["node"], agent["status"]) for agent in history.agents(2)] == [
        (2, "stopped")
    ]
    assert history.agents(1)[0]["task"] == "Survey CS."


def test_events_naming_a_node_or_cell_no_run_could_have_are_left_out(library):
    plant(library, CHAT, recorded("nested"))
    run = crafted(
        *ROOT_AND_CELL,
        {"kind": "agent.start", "node": [2], "parent": 1, "cell": 1},
        {"kind": "usage", "node": [2], "cost": 0.5},
        {"kind": "cell.start", "node": 1, "cell": {"id": 2}, "code": "print('out')"},
    )
    plant(library, CHAT, run)
    history = history_of(library)
    with capture_logs() as logs:
        turns = history.turns()
    assert [turn["message"] for turn in turns] == ["Plan the term.", "Crafted."]
    assert history.agents(2) == []
    assert [step["code"] for step in history.steps(2)] == ["print('kept')"]
    left_out = [
        entry["seq"] for entry in logs if entry["event"] == "history.event_left_out"
    ]
    assert left_out == [4, 5, 6]


def test_the_earlier_runs_are_read_once_and_handed_out_as_copies(library):
    plant(library, CHAT, recorded("nested"))
    history = history_of(library)
    history.steps(1)[0]["code"] = "changed"
    plant(library, CHAT, recorded("fanout"))
    assert [turn["message"] for turn in history.turns()] == ["Plan the term."]
    assert history.steps(1)[0]["code"] != "changed"


def test_a_call_with_an_unpriced_model_has_no_cost(library):
    plant(library, CHAT, recorded("unpriced"))
    history = history_of(library)
    assert history.turns()[0]["cost"] is None
    assert [agent["cost"] for agent in history.agents(1)] == [None, None]


def test_a_message_or_node_that_is_not_there_is_refused_in_plain_words(library):
    plant(library, CHAT, recorded("nested"))
    history = history_of(library)
    for call in (lambda: history.steps(2), lambda: history.agents(0)):
        with pytest.raises(ValueError, match="numbered 1 to 1"):
            call()
    with pytest.raises(ValueError, match=r"nodes are \[1, 2, 3\]"):
        history.steps(1, node=7)
    with pytest.raises(ValueError, match="no earlier messages"):
        history_of(library, ANOTHER_CHAT).agents(1)


MODEL_DOWN = (
    "InternalServerError: Error code: 500 - {'error': {'message': 'the fake model is"
    " down'}}"
)


def test_errors_list_each_failed_cell_the_top_agents_included_and_each_failed_agent(
    library,
):
    plant(library, CHAT, recorded("nested"))
    plant(library, CHAT, recorded("failing"))
    history = history_of(library)
    top = {"turn": 2, "node": None, "namespace": "root", "task": "Survive failures."}
    assert history.errors(1) == []
    assert history.errors(2) == [
        top
        | {"index": 1, "error": "ZeroDivisionError: division by zero"}
        | {"code": "print(1 / 0)"},
        top
        | {"index": 2, "error": f"openai.{MODEL_DOWN}"}
        | {"code": "r = subagent('Break the model.')\nprint(r)"},
        {"turn": 2, "node": 2, "namespace": "root", "task": "Break the model."}
        | {"index": None}
        | {"error": f"It ended failed: {MODEL_DOWN}", "code": None},
    ]
    assert history.errors() == history.errors(2)
    for error in history.errors(2)[:2]:
        assert history.steps(2)[error["index"]]["code"] == error["code"]


@pytest.mark.parametrize(
    ("name", "top", "sub_agent"),
    [
        ("stopped", "It ended stopped.", "It ended stopped."),
        ("exhausted", f"It ended exhausted: {EXHAUSTED}", "It ended exhausted."),
    ],
)
def test_errors_list_each_agent_that_stopped_or_ran_out_of_turns(
    library, name, top, sub_agent
):
    plant(library, CHAT, recorded(name))
    assert [(each["node"], each["error"]) for each in history_of(library).errors()] == [
        (None, top),
        (2, sub_agent),
    ]


def test_errors_list_a_run_the_app_stopped_under_as_failed(library):
    plant(library, CHAT, recorded("killed")[:-1])
    top, _ = history_of(library).errors(1)
    assert (top["node"], top["error"]) == (None, f"It ended failed: {LOST}")


def test_errors_give_what_a_sub_agents_cell_raised_and_its_index_in_its_steps(library):
    plant(library, CHAT, recorded("raising"))
    history = history_of(library)
    found = history.errors(1)
    assert [(each["node"], each["index"], each["error"]) for each in found] == [
        (2, 1, "ValueError: 2 problems:\ntools: bad\nspawn: bad"),
        (2, 2, "ExceptionGroup: x (2 sub-exceptions)\nValueError: 1\nKeyError: 2"),
        (2, 3, "ValueError: after"),
    ]
    assert {each["task"] for each in found} == {"Raise them."}
    for each in found:
        assert history.steps(1, node=2)[each["index"]]["code"] == each["code"]


@pytest.mark.parametrize("outcome", [["failed"], {"failed": True}])
def test_a_crafted_outcome_that_is_no_text_leaves_errors_and_find_working(
    library, outcome
):
    run = crafted(*ROOT_AND_CELL)
    run[-1] |= {"outcome": outcome}
    plant(library, CHAT, run)
    history = history_of(library)
    assert history.errors() == []
    assert [each["kind"] for each in history.find("crafted")] == ["task", "result"]


def test_errors_and_find_read_a_crafted_record_whose_fields_are_no_text(library):
    run = crafted(
        {"kind": "agent.start", "node": 1, "parent": None, "cell": None},
        {"kind": "think", "node": 1, "text": ["odd"]},
        {"kind": "cell.start", "node": 1, "cell": 1, "code": {"odd": "code"}},
        {"kind": "cell.end", "node": 1, "cell": 1, "code": 7, "output": ["odd"]}
        | {"status": "failed"},
        {"kind": "agent.start", "node": 2, "parent": 1, "cell": 1, "task": ["odd"]},
        {"kind": "agent.end", "node": 2, "status": ["failed"], "result": {"odd": 1}},
        {"kind": "agent.start", "node": 3, "parent": 1, "cell": 1, "task": None},
        {"kind": "agent.end", "node": 3, "status": "failed", "result": 5},
    )
    plant(library, CHAT, run)
    history = history_of(library)
    assert history.errors(1) == [
        {"turn": 1, "node": None, "namespace": None, "task": "Crafted.", "index": 1}
        | {"error": None, "code": 7},
        {"turn": 1, "node": 3, "namespace": None, "task": None, "index": None}
        | {"error": "It ended failed: 5", "code": None},
    ]
    assert history.find("odd") == []
    assert [each["kind"] for each in history.find("crafted")] == ["task", "result"]


# Deeper than Python recurses by default, as a file from someone else may nest a value.
DEEP = json.loads("[" * 1_200 + '"deep"' + "]" * 1_200)


def test_a_value_nested_too_deep_is_marked_where_it_nests_too_deep_and_prints(library):
    run = crafted(
        *ROOT_CELL_AND_SUB_AGENT,
        {"kind": "agent.end", "node": 2, "status": "failed", "result": DEEP},
        {"kind": "cell.end", "node": 1, "cell": 1, "code": DEEP, "output": DEEP}
        | {"status": "failed"},
    )
    run[-1] |= {"answer": DEEP}
    plant(library, CHAT, run)
    history = history_of(library)
    for listed in (
        history.turns(),
        history.agents(1),
        history.steps(1),
        history.errors(1),
    ):
        assert NESTED_TOO_DEEP in str(listed)
    code = history.steps(1)[0]["code"]
    # Under the list of steps and the cell's dict.
    for _ in range(MAX_VALUE_DEPTH - 2):
        [code] = code
    assert code == NESTED_TOO_DEEP


def test_full_refuses_a_step_nested_too_deep_to_copy_in_words(library):
    run = crafted(
        *ROOT_AND_CELL,
        {"kind": "cell.end", "node": 1, "cell": 1, "code": DEEP, "output": "kept"}
        | {"status": "done"},
    )
    plant(library, CHAT, run)
    with pytest.raises(ValueError, match=re.escape(TOO_DEEP)):
        history_of(library).full(1, None, 0)


@pytest.mark.parametrize("node", [None, 1])
def test_a_crafted_sub_agent_named_as_the_top_agent_is_never_listed_as_it(
    library, node
):
    run = crafted(
        *ROOT_AND_CELL,
        {"kind": "agent.start", "node": node, "parent": 1, "cell": 1, "task": "Odd."},
        {"kind": "cell.start", "node": node, "cell": 2, "code": "odd()"},
        {"kind": "cell.end", "node": node, "cell": 2, "code": "odd()"}
        | {"output": "NameError: odd", "status": "failed"},
    )
    plant(library, CHAT, run)
    history = history_of(library)
    assert history.errors(1) == []
    assert history.find("odd") == []
    assert [step["code"] for step in history.steps(1)] == ["print('kept')"]


def test_errors_and_find_list_at_most_50_rows_then_how_many_more(library):
    cells = [
        each
        for cell in range(1, MAX_FOUND + 11)
        for each in (
            {"kind": "cell.start", "node": 1, "cell": cell, "code": "fail()"},
            {"kind": "cell.end", "node": 1, "cell": cell, "code": "fail()"}
            | {"output": "NameError: name 'f' is not defined", "status": "failed"},
        )
    ]
    plant(library, CHAT, crafted(ROOT_AND_CELL[0], *cells))
    history = history_of(library)
    errors, found = history.errors(), history.find("fail")
    assert [len(errors), len(found)] == [MAX_FOUND + 1, MAX_FOUND + 1]
    assert errors[MAX_FOUND - 1]["index"] == found[MAX_FOUND - 1]["index"] == 49
    assert errors[-1] == {
        "more": 10,
        "note": "10 more not listed: pass a turn to list only that message's.",
    }
    assert found[-1] == {
        "more": 10,
        "note": "10 more not listed: pass a turn, or longer text, to list fewer.",
    }
    assert f"at most {MAX_FOUND} rows" in HISTORY_DESCRIPTION


def test_turns_and_agents_count_the_failed_cells_the_turns_its_sub_agents_too(library):
    plant(library, CHAT, recorded("nested"))
    plant(library, CHAT, recorded("failing"))
    history = history_of(library)
    assert [turn["failed"] for turn in history.turns()] == [0, 2]
    assert [agent["failed"] for agent in history.agents(2)] == [0]


def test_find_looks_through_every_agents_tasks_code_outputs_and_results_ignoring_case(
    library,
):
    plant(library, CHAT, recorded("nested"))
    plant(library, CHAT, recorded("failing"))
    history = history_of(library)
    found = history.find("cs101")
    assert [(each["node"], each["index"], each["kind"]) for each in found] == [
        (None, 0, "output"),
        (None, 1, "output"),
        (None, None, "result"),
        (2, 0, "code"),
        (2, 0, "output"),
        (2, 1, "output"),
        (2, None, "result"),
        (3, None, "task"),
        (3, 0, "code"),
        (3, 0, "output"),
        (3, None, "result"),
    ]
    assert {each["turn"] for each in found} == {1}
    assert found[0]["snippet"] == "CS: CS101 is light"
    [thought] = history.find("THIS CELL", turn=2)
    assert (thought["node"], thought["index"], thought["kind"]) == (None, 0, "think")
    assert history.find("CS101", turn=2) == []


def test_find_shows_a_snippet_around_the_match(library):
    long = "x" * 100 + "Needle" + "y" * 100
    found = {"kind": "cell.end", "node": 1, "cell": 1, "code": "", "output": long}
    plant(library, CHAT, crafted(*ROOT_AND_CELL, found | {"status": "done"}))
    [match] = history_of(library).find("needle")
    assert match["snippet"] == f"…{'x' * 60}Needle{'y' * 60}…"


@pytest.mark.parametrize("text", ["", 5, None])
def test_find_takes_text(library, text):
    plant(library, CHAT, recorded("nested"))
    with pytest.raises(ValueError, match=re.escape(NOT_FINDABLE)):
        history_of(library).find(text)


def test_errors_and_find_in_a_chat_with_no_earlier_messages_say_so(library):
    plant(library, ANOTHER_CHAT, recorded("failing"))
    history = history_of(library)
    for call in (history.errors, lambda: history.find("x")):
        with pytest.raises(ValueError, match="no earlier messages"):
            call()


LONG = "7" * (MAX_VALUE_CHARS + 500)
CUT = f"{'7' * MAX_VALUE_CHARS}…(+500 chars)"


def test_history_cuts_each_long_text_and_full_gives_a_step_whole(library):
    run = crafted(
        *ROOT_CELL_AND_SUB_AGENT,
        {"kind": "agent.end", "node": 2, "status": "done", "result": LONG},
        {"kind": "cell.end", "node": 1, "cell": 1, "code": LONG, "output": LONG}
        | {"status": "failed"},
    )
    plant(library, CHAT, run)
    history = history_of(library)
    [cell] = history.steps(1)
    [error] = history.errors(1)
    assert (cell["code"], cell["output"]) == (CUT, CUT)
    assert history.agents(1)[0]["result"] == CUT
    assert (error["error"], error["code"]) == (CUT, CUT)
    assert {each["snippet"] for each in history.find("7")} == {f"7{'7' * 60}…"}
    assert history.full(1, None, 0)["output"] == LONG
    assert history.full(1, node=None, index=-1)["code"] == LONG
    assert f"{MAX_VALUE_CHARS:,} characters" in HISTORY_DESCRIPTION
    p = page_of(library).new("Whole")
    p.step(1, index=0)
    assert saved(library, p)[0]["step"]["output"] == LONG


def test_full_refuses_a_step_that_isnt_there_in_words(library):
    plant(library, CHAT, recorded("nested"))
    with pytest.raises(ValueError, match="top agent has 2 steps, numbered from 0"):
        history_of(library).full(1, None, 2)
    with pytest.raises(ValueError, match=re.escape(NOT_AN_INDEX)):
        history_of(library).full(1, None, "0")


def test_the_tool_reads_the_app_file_without_writing_it_or_running_its_own_sql(
    library,
):
    plant(library, CHAT, recorded("nested"))
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute("CREATE VIEW planted AS SELECT id FROM chats")
        connection.execute(
            "CREATE TRIGGER wipe AFTER INSERT ON events BEGIN DELETE FROM runs; END"
        )
    before = library.read_bytes()
    with closing(open_existing(library)) as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM events")
        with pytest.raises(sqlite3.OperationalError, match="planted"):
            connection.execute("SELECT * FROM planted")
    assert history_of(library).turns()[0]["message"] == "Plan the term."
    assert library.read_bytes() == before


EXAMPLE = """name: count the tickets
messages:
- role: user
  content: How many tickets are open?
"""


# EXAMPLE holds only the task, so as a worked example it is ✗.
ONLY_THE_TASK = [
    "It ends on turn 1 (user); end with the assistant's turn that calls FinalAnswer(...)."
]
NOT_THIS_CHAT = "The Library can be changed only in a chat whose namespace grants"
SHELL_KEY_ENV = "A_PROVIDERS_KEY"
SHELL_KEY = "sk-a-key-from-the-shell"
# The key in YAML's double quotes with its "s" escaped: the text never holds it, the data does.
ESCAPED_KEY = '"\\x73' + SHELL_KEY[1:] + '"'


# The engine config of the run the library tool is in, as its spec.yaml holds it: a
# model on a server nothing calls, as no check calls one.
RUN_CONFIG = {
    "model": "gpt-6-luna",
    "client": {"base_url": "http://127.0.0.1:9/v1", "api_key_env": "A_TEST_MODEL_KEY"},
}


def library_func(library, namespace="librarian"):
    """The library tool, with what its agents are told, as a run of a chat in *namespace*
    of a fresh Library gets it, in a run folder with its spec."""
    run = library.parent / "run"
    run.mkdir(exist_ok=True)
    (run / SPEC).write_text(yaml.safe_dump({"config": RUN_CONFIG}))
    told = {
        "library": str(library),
        "chat": CHAT,
        "run": CURRENT,
        "tools_dir": str(run / "tools"),
        "namespace": namespace,
        "key_names": [SHELL_KEY_ENV],
        "editors": [LIBRARIAN],
    }
    return build_library(None, told)


def library_tool(library, namespace="librarian"):
    return library_func(library, namespace).value


# A fresh app file's decompositions: root's hand-offs to the librarian, then to the
# historian, then the historian's examples of the same names, then the librarian's.
FRESH = [
    *[("root", parsed(text).name) for text in LIBRARIAN_HAND_OFFS],
    *[
        (namespace, parsed(text).name)
        for namespace in ("root", HISTORIAN)
        for text in HISTORIANS_EXAMPLES
    ],
    *[(LIBRARIAN, parsed(text).name) for text in LIBRARIANS_EXAMPLES],
]


def stored(library):
    """Each decomposition in the app file, as (namespace, name)."""
    with closing(open_library(library)) as connection:
        return connection.execute(
            "SELECT namespace, name FROM decompositions ORDER BY rowid"
        ).fetchall()


@pytest.fixture
def fresh(library):
    """A fresh app file, with a namespace drafts beside the librarian."""
    with closing(open_library(library)) as connection:
        add_namespace(connection, "drafts", "")
    return library


def test_show_lists_every_namespace_in_brief_as_plain_data(fresh):
    tool = library_tool(fresh)
    done = tool.write_decomposition("drafts", EXAMPLE)
    shown = tool.show()
    by_name = {each["name"]: each for each in shown}
    assert done == {
        "done": "added",
        "what": "decomposition",
        "namespace": "drafts",
        "name": "count the tickets",
        "problems": ONLY_THE_TASK,
    }
    assert json.loads(json.dumps(shown)) == shown
    assert list(by_name) == ["root", "drafts", HISTORIAN, LIBRARIAN]
    assert by_name["drafts"] == {
        "name": "drafts",
        "depth": 1,
        "sets": {},
        "tools": ["llm"],
        "decompositions": [{"name": "count the tickets", "problems": 1}],
    }
    assert by_name[LIBRARIAN]["sets"]["spawn"] == [HISTORIAN]
    assert by_name[LIBRARIAN]["tools"] == ["llm", "library"]
    assert by_name[LIBRARIAN]["decompositions"] == [
        {"name": parsed(text).name, "problems": 0} for text in LIBRARIANS_EXAMPLES
    ]


def test_show_of_a_name_gives_that_namespace_whole_as_the_panel_shows_it(fresh):
    tool = library_tool(fresh)
    tool.write_decomposition("drafts", EXAMPLE)
    drafts = tool.show("drafts")
    assert (drafts["name"], drafts["depth"], drafts["yaml"]) == ("drafts", 1, "")
    assert drafts["decompositions"] == [
        {"name": "count the tickets", "yaml": EXAMPLE, "problems": ONLY_THE_TASK}
    ]
    assert {each["label"] for each in drafts["fields"]} >= {"Tools", "System suffix"}


def test_show_cuts_each_long_text_in_its_brief_list_but_not_for_one_namespace(fresh):
    suffix = "Cite. " * 1_000
    tool = library_tool(fresh)
    tool.change_namespace("drafts", {"system_suffix": suffix})
    [drafts] = [each for each in tool.show() if each["name"] == "drafts"]
    cut_at = BRIEF_CHARS
    assert drafts["sets"]["system_suffix"] == (
        f"{suffix[:cut_at]}…(+{len(suffix) - cut_at:,} chars)"
    )
    assert yaml.safe_load(tool.show("drafts")["yaml"]) == {"system_suffix": suffix}


@pytest.mark.parametrize(
    ("method", "refusal"),
    [
        ("show", 'The Library has no namespace "nowhere": show() lists them.'),
        ("tools", 'The Library has no tool "nowhere": tools() lists them.'),
    ],
)
def test_reading_one_that_isnt_there_says_where_they_are_listed(fresh, method, refusal):
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        getattr(library_tool(fresh), method)("nowhere")


@pytest.mark.parametrize(
    ("text", "problems"),
    [(HISTORIANS_EXAMPLES[0], []), (EXAMPLE, ONLY_THE_TASK)],
    ids=["worked-example", "only-the-task"],
)
def test_a_written_decomposition_is_stored_and_the_result_gives_its_problems(
    fresh, text, problems
):
    done = library_tool(fresh).write_decomposition("drafts", text)
    assert done["problems"] == problems
    assert ("drafts", done["name"]) in stored(fresh)


@pytest.mark.parametrize(
    ("content", "stored_content"),
    [
        (
            "|\n    [repl]\n    FinalAnswer(3)\n    [/repl]\n",
            "|\n    <repl>\n    FinalAnswer(3)\n    </repl>\n",
        ),
        ("[repl]FinalAnswer(3)[/repl]\n", "<repl>FinalAnswer(3)</repl>\n"),
    ],
    ids=["block", "one-line"],
)
def test_a_decomposition_written_with_repl_placeholders_is_stored_with_the_real_tags(
    fresh, content, stored_content
):
    answer = "- role: assistant\n  content: "
    tool = library_tool(fresh)
    tool.write_decomposition("drafts", EXAMPLE + answer + content)
    assert tool.show("drafts")["decompositions"] == [
        {
            "name": "count the tickets",
            "yaml": EXAMPLE + answer + stored_content,
            "problems": ["Turn 2 (assistant) has no <think>…</think>."],
        }
    ]


class NeverEqual(str):
    """Text that compares unequal to everything, itself included."""

    def __eq__(self, other):
        return False

    __hash__ = str.__hash__


@pytest.mark.parametrize(
    ("namespace", "text", "refusal"),
    [
        (NeverEqual("root"), EXAMPLE, NOT_TEXT),
        ("drafts", NeverEqual(EXAMPLE), NOT_TEXT),
        ("root", EXAMPLE, "root is locked"),
        ("drafts", "#" * (MAX_YAML_CHARS + 1), "may be at most 1,000,000 characters"),
        ("drafts", EXAMPLE + f"# {SHELL_KEY}\n", "holds a provider's key"),
        (
            "drafts",
            EXAMPLE.replace("How many tickets are open?", ESCAPED_KEY),
            "holds a provider's key",
        ),
        (
            "drafts",
            EXAMPLE.replace("How many tickets are open?", '"{{ open }} open?"'),
            (
                "Turn 1 (user), line 1: the engine reads a turn holding {{ or {% as a Jinja"
                ' template, and this one names "open"'
            ),
        ),
    ],
    ids=[
        "text-unequal-to-root",
        "yaml-not-plain-text",
        "root",
        "too-long",
        "a-key",
        "an-escaped-key",
        "a-template-the-engine-cant-render",
    ],
)
def test_a_write_the_tool_refuses_says_why_and_stores_nothing(
    fresh, monkeypatch, namespace, text, refusal
):
    monkeypatch.setenv(SHELL_KEY_ENV, SHELL_KEY)
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        library_tool(fresh).write_decomposition(namespace, text)
    assert stored(fresh) == FRESH


@pytest.mark.parametrize(
    ("change", "argument"),
    [
        (change_namespace, "tools: [llm]\n"),
        (delete_namespace, len(LIBRARIANS_EXAMPLES)),
    ],
)
def test_a_chat_whose_namespace_lost_the_grant_mid_run_may_not_write(
    fresh, change, argument
):
    tool = library_tool(fresh)
    with closing(open_library(fresh)) as connection:
        change(connection, "librarian", argument)
    before = stored(fresh)
    with pytest.raises(LibraryError, match=NOT_THIS_CHAT):
        tool.write_decomposition("drafts", EXAMPLE)
    assert stored(fresh) == before


def test_the_tools_agents_are_told_they_may_only_read_in_a_chat_that_cant_write(
    fresh,
):
    assert library_func(fresh).description == LIBRARY_DESCRIPTION
    told = library_func(fresh, "drafts").description
    assert told.startswith(LIBRARY_DESCRIPTION)
    assert told.endswith(
        "\nIn this chat you may only read the Library: its namespace, drafts, doesn't"
        " grant library, so the tool refuses every write, here and in any sub-agent."
        " Asked for a change, don't try it: answer, with FinalAnswer, what you would"
        " change, and that the user picks librarian under the message box and asks there."
    )


def test_a_busy_app_file_is_a_refusal_in_words(fresh):
    refused = pytest.raises(LibraryError, match="can't be used now .*; try again")
    with closing(open_library(fresh)) as connection, writing(connection), refused:
        library_tool(fresh).write_decomposition("drafts", EXAMPLE)
    assert stored(fresh) == FRESH


def test_a_moved_app_file_is_never_made_afresh(fresh, tmp_path):
    tool = library_tool(fresh)
    fresh.rename(tmp_path / "moved.sqlite")
    for call in (tool.show, lambda: tool.write_decomposition("drafts", EXAMPLE)):
        with pytest.raises(LibraryError, match="can't be used now"):
            call()
    assert not fresh.exists()


def stored_namespaces(library):
    """Each namespace in the app file, as (name, yaml)."""
    with closing(open_library(library)) as connection:
        return connection.execute("SELECT name, yaml FROM namespaces").fetchall()


@pytest.mark.parametrize("method", ["add_namespace", "change_namespace"])
@pytest.mark.parametrize(
    ("name", "text"),
    [(NeverEqual("root"), ""), ("drafts", NeverEqual("tools: [history]\n"))],
    ids=["name-unequal-to-root", "yaml-not-plain-text"],
)
def test_a_namespace_write_takes_only_plain_text(fresh, method, name, text):
    before = stored_namespaces(fresh)
    with pytest.raises(LibraryError, match=re.escape(NOT_TEXT)):
        getattr(library_tool(fresh), method)(name, text)
    assert stored_namespaces(fresh) == before


@pytest.mark.parametrize(
    ("method", "name"),
    [
        ("add_namespace", "research"),
        ("change_namespace", "drafts"),
        ("add_namespace", "drafts"),
        ("add_namespace", "missing.child"),
        ("change_namespace", "nowhere"),
    ],
    ids=["add", "change", "add-existing", "add-under-missing", "change-missing"],
)
def test_a_chat_whose_namespace_doesnt_grant_the_tool_may_not_write_a_namespace(
    fresh, method, name
):
    before = stored_namespaces(fresh)
    with pytest.raises(LibraryError, match=NOT_THIS_CHAT):
        getattr(library_tool(fresh, "drafts"), method)(name, "system_suffix: Hi.\n")
    assert stored_namespaces(fresh) == before


def own_fields(library, name):
    """The fields namespace *name* sets itself, in the app file."""
    return yaml.safe_load(dict(stored_namespaces(library))[name]) or {}


def test_a_change_keeping_the_repl_and_app_tools_the_user_set_is_written(fresh):
    set_by_the_user = "tools: [history]\nrepl: {type: local}\n"
    changed = "repl: {type: local}\ntools: [llm, history]\nsystem_suffix: Draft.\n"
    with closing(open_library(fresh)) as connection:
        change_namespace(connection, "drafts", set_by_the_user)
    done = library_tool(fresh).change_namespace("drafts", changed)
    assert done == {
        "done": "changed",
        "what": "namespace",
        "namespace": "drafts",
        "name": "drafts",
        "changed": {
            "tools": {"before": ["history"], "after": ["llm", "history"]},
            "system_suffix": {"before": None, "after": "Draft."},
        },
        "kept": ["repl"],
    }
    assert own_fields(fresh, "drafts") == yaml.safe_load(changed)


RESEARCHING = "system_suffix: Cite a source.\ntools: [llm]\nvars: {year: 2026}\n"


@pytest.mark.parametrize(
    ("given", "fields"),
    [
        (
            "tools: []\n",
            {"system_suffix": "Cite a source.", "tools": [], "vars": {"year": 2026}},
        ),
        (
            "system_suffix: null\nvars: {year: 2027}\n",
            {"tools": ["llm"], "vars": {"year": 2027}},
        ),
        ({"tools": None, "vars": None}, {"system_suffix": "Cite a source."}),
    ],
    ids=["one-field", "a-field-set-to-null-is-taken-out", "a-dict"],
)
def test_a_change_sets_only_the_fields_it_gives_and_keeps_the_rest(
    fresh, given, fields
):
    tool = library_tool(fresh)
    tool.add_namespace("research", RESEARCHING)
    tool.change_namespace("research", given)
    assert own_fields(fresh, "research") == fields


def test_a_change_with_replace_sets_every_field_as_the_panels_edit_yaml_does(fresh):
    tool = library_tool(fresh)
    tool.add_namespace("research", RESEARCHING)
    done = tool.change_namespace("research", "spawn: [historian]\n", replace=True)
    assert ("research", "spawn: [historian]\n") in stored_namespaces(fresh)
    assert done["changed"] == {
        "system_suffix": {"before": "Cite a source.", "after": None},
        "tools": {"before": ["llm"], "after": None},
        "vars": {"before": {"year": 2026}, "after": None},
        "spawn": {"before": None, "after": ["historian"]},
    }
    assert done["kept"] == []


def test_a_change_that_changes_nothing_says_so_and_keeps_the_yaml_as_typed(fresh):
    commented = "# kept as typed\nsystem_suffix: Cite a source.\n"
    tool = library_tool(fresh)
    tool.add_namespace("research", commented)
    done = tool.change_namespace("research", {"system_suffix": "Cite a source."})
    assert done == {
        "done": "unchanged",
        "what": "namespace",
        "namespace": "research",
        "name": "research",
    }
    assert ("research", commented) in stored_namespaces(fresh)


@pytest.mark.parametrize(
    ("method", "args", "stored_yaml"),
    [
        ("add_namespace", ("research", {"tools": ["llm"]}), "tools:\n- llm\n"),
        (
            "add_namespace",
            ("research", "\n\nsystem_suffix: Hi.\n"),
            "system_suffix: Hi.\n",
        ),
    ],
    ids=["a-dict", "leading-blank-lines"],
)
def test_a_namespace_is_stored_as_clean_yaml(fresh, method, args, stored_yaml):
    getattr(library_tool(fresh), method)(*args)
    assert ("research", stored_yaml) in stored_namespaces(fresh)


def test_a_decomposition_may_be_written_as_a_dict(fresh):
    as_dict = yaml.safe_load(EXAMPLE)
    done = library_tool(fresh).write_decomposition("drafts", as_dict)
    assert (done["name"], done["problems"]) == ("count the tickets", ONLY_THE_TASK)
    assert ("drafts", "count the tickets") in stored(fresh)


@pytest.mark.parametrize(
    "given", [["tools"], {"tools": {1, 2}}, {1: "x"}], ids=["list", "a-set", "int-key"]
)
def test_yaml_given_as_anything_but_text_or_a_dict_of_plain_values_is_refused(
    fresh, given
):
    before = stored_namespaces(fresh)
    with pytest.raises(LibraryError, match=re.escape(NOT_TEXT)):
        library_tool(fresh).change_namespace("drafts", given)
    assert stored_namespaces(fresh) == before


def test_a_decomposition_whose_code_is_not_python_says_so(fresh):
    # As a cell's string without r writes an escape the example's code held: a line break.
    broken = (
        EXAMPLE
        + "- role: assistant\n  content: |\n    <think>Count.</think>\n"
        + '    [repl]\n    FinalAnswer("one\n    two")\n    [/repl]\n'
    )
    done = library_tool(fresh).write_decomposition("drafts", broken)
    assert done["problems"] == []
    assert done["code_problems"] == [
        (
            "Turn 2 (assistant)'s code is not valid Python (line 1: unterminated string"
            " literal (detected at line 1))."
        )
    ]


@pytest.mark.parametrize(
    ("root", "name", "level"),
    [
        ("tools: [llm, history]\n", "librarian.notes", "librarian"),
        ("tools: [llm, history, library]\n", "research", "root"),
    ],
    ids=["under-librarian", "under-a-root-granting-library"],
)
def test_a_namespace_the_tool_adds_never_inherits_library(fresh, root, name, level):
    with closing(open_library(fresh)) as connection:
        change_namespace(connection, "root", root)
    before = stored_namespaces(fresh)
    refusal = f"{name} would inherit library from {level}: which namespaces grant"
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        library_tool(fresh).add_namespace(name, "")
    assert stored_namespaces(fresh) == before


@pytest.fixture
def coder(fresh):
    """A fresh app file whose Library has coder, a tool of the user's that runs Claude
    Code."""
    with closing(open_library(fresh)) as connection, writing(connection):
        connection.execute(
            UPSERT_TOOL, ("coder", dump({"factory": "claude_code"}), None)
        )
    return fresh


@pytest.mark.parametrize(
    ("method", "name"), [("add_namespace", "research"), ("change_namespace", "drafts")]
)
def test_a_tool_that_runs_claude_code_is_never_granted_afresh(coder, method, name):
    before = stored_namespaces(coder)
    refusal = (
        "tools may not add coder, which runs Claude Code on the user's subscription"
    )
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        getattr(library_tool(coder), method)(name, "tools: [llm, coder]\n")
    assert stored_namespaces(coder) == before


@pytest.mark.parametrize(
    "changed", ["tools: [coder]\nsystem_suffix: Draft.\n", "tools: []\n"]
)
def test_a_claude_code_tool_the_user_granted_may_be_kept_or_taken_out(coder, changed):
    with closing(open_library(coder)) as connection:
        change_namespace(connection, "drafts", "tools: [coder]\n")
    assert library_tool(coder).change_namespace("drafts", changed)["done"] == "changed"
    assert own_fields(coder, "drafts") == yaml.safe_load(changed)


@pytest.mark.parametrize(
    ("method", "name"), [("add_namespace", "research"), ("change_namespace", "drafts")]
)
def test_a_namespace_whose_yaml_spells_a_key_with_an_escape_is_refused(
    fresh, monkeypatch, method, name
):
    monkeypatch.setenv(SHELL_KEY_ENV, SHELL_KEY)
    before = stored_namespaces(fresh)
    with pytest.raises(LibraryError, match="holds a provider's key"):
        getattr(library_tool(fresh), method)(name, f"system_suffix: {ESCAPED_KEY}\n")
    assert stored_namespaces(fresh) == before


@pytest.mark.parametrize(
    ("parent", "own", "refusal"),
    [
        (
            "coding",
            "reasoner: {type: claude_code}\n",
            "coding.x would inherit the claude_code reasoner from coding, which runs on",
        ),
        (
            "coding",
            "tools: [coder]\n",
            "coding.x would inherit coder, which runs Claude Code on the user's",
        ),
    ],
    ids=["a-claude-code-reasoner", "a-claude-code-tool"],
)
def test_a_namespace_the_tool_adds_never_inherits_claude_code(
    coder, parent, own, refusal
):
    with closing(open_library(coder)) as connection:
        add_namespace(connection, parent, own)
    before = stored_namespaces(coder)
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        library_tool(coder).add_namespace(f"{parent}.x", "")
    assert stored_namespaces(coder) == before


def stored_tools(library):
    """Each tool in the app file, as (name, yaml, source)."""
    with closing(open_library(library)) as connection:
        return connection.execute("SELECT name, yaml, source FROM tools").fetchall()


# A factory that raises: a refusal other than the build's shows it never ran.
RAISING = "def build(client, params):\n    raise RuntimeError('built')\n"


def test_a_chat_whose_namespace_doesnt_grant_the_tool_may_not_write_a_tool_nor_build_it(
    fresh,
):
    with pytest.raises(LibraryError, match=NOT_THIS_CHAT):
        library_tool(fresh, "drafts").write_tool("counter", RAISING, "build")
    assert stored_tools(fresh) == []


def test_a_chat_whose_namespace_runs_a_sandboxed_repl_may_not_write_a_tool_nor_build_it(
    fresh,
):
    sandboxed = "tools: [library]\nrepl: {type: restricted}\n"
    with closing(open_library(fresh)) as connection:
        change_namespace(connection, "librarian", sandboxed)
    with pytest.raises(LibraryError, match=re.escape(NOT_LOCAL)):
        library_tool(fresh).write_tool("counter", RAISING, "build")
    assert stored_tools(fresh) == []


HOLDS_ITSELF = []
HOLDS_ITSELF.append(HOLDS_ITSELF)


@pytest.mark.parametrize(
    ("args", "refusal"),
    [
        ((NeverEqual("counter"), WORD_COUNT, "build"), NOT_A_TOOL),
        (("counter", WORD_COUNT, "build", {"sep": NeverEqual(",")}), NOT_A_TOOL),
        (("counter", WORD_COUNT, "build", {"sep": (",",)}), NOT_A_TOOL),
        (("counter", WORD_COUNT, "build", {1: ","}), NOT_A_TOOL),
        (("counter", WORD_COUNT, "build", {"seps": HOLDS_ITSELF}), NOT_A_TOOL),
        (("counter", WORD_COUNT, "build", [","]), NOT_A_TOOL),
        (("counter", WORD_COUNT, "build", {"factory_from": "/x.py"}), SETS_OWN_KEYS),
        (
            ("counter", f"{WORD_COUNT}# {SHELL_KEY}\n", "build"),
            "holds a provider's key",
        ),
        (
            ("counter", f"{WORD_COUNT}KEY = {ESCAPED_KEY}\n", "build"),
            "holds a provider's",
        ),
        (
            ("counter", WORD_COUNT, "build", {"key": SHELL_KEY}),
            "holds a provider's key",
        ),
        (
            ("counter", f"{WORD_COUNT}KEY = b{ESCAPED_KEY}\n", "build"),
            "holds a provider's key",
        ),
        (
            ("counter", f"x = {'1+' * 300_000}1\n", "build"),
            "The source is nested too deeply to parse.",
        ),
    ],
    ids=[
        "name-not-plain-text",
        "a-setting-not-plain-text",
        "a-tuple-setting",
        "a-setting-not-keyed-by-text",
        "a-setting-holding-itself",
        "settings-not-a-dict",
        "settings-setting-factory-from",
        "a-key-in-the-source",
        "a-key-spelled-by-an-escape",
        "a-key-in-the-settings",
        "a-key-in-a-bytes-literal",
        "a-source-too-deep-to-parse",
    ],
)
def test_a_tool_the_tool_refuses_says_why_and_stores_nothing(
    fresh, monkeypatch, args, refusal
):
    monkeypatch.setenv(SHELL_KEY_ENV, SHELL_KEY)
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        library_tool(fresh).write_tool(*args)
    assert stored_tools(fresh) == []


def test_a_tool_the_engine_builds_from_its_block_alone_is_never_replaced(coder):
    before = stored_tools(coder)
    refusal = (
        "coder is built by the engine's own claude_code factory from its block alone,"
        " so it has no Python for write_tool to write back: only the user may replace"
        " it, by importing a config that defines it (deep-reasoning import)."
    )
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        library_tool(coder).write_tool("coder", WORD_COUNT, "build")
    assert stored_tools(coder) == before


def test_a_tool_root_grants_is_never_replaced(fresh):
    tool = library_tool(fresh)
    tool.write_tool("search", WORD_COUNT, "build")
    with closing(open_library(fresh)) as connection:
        change_namespace(connection, "root", "tools: [llm, history, search]\n")
    before = stored_tools(fresh)
    refusal = (
        "root grants search, so every chat would run the code you write: only the user"
        " may replace it, once they take it out of root's tools in the Library panel's"
        " Namespaces tab (Edit YAML)."
    )
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        tool.write_tool("search", RAISING, "build")
    assert stored_tools(fresh) == before


def test_a_tool_is_built_from_a_file_deleted_after_the_build(fresh):
    tool = library_tool(fresh)
    tool.write_tool("counter", WORD_COUNT, "build")
    with pytest.raises(LibraryError, match="raised RuntimeError: built"):
        tool.write_tool("other", RAISING, "build")
    assert list((fresh.parent / "run" / "tools" / "checks").iterdir()) == []


HELPED = """from deep_reasoner.primitives import Func
from helper import count


def build(client, params):
    return Func(count)
"""
READS_A_FILE = """from pathlib import Path

from deep_reasoner.primitives import Func


def build(client, params):
    words = Path("words.txt").read_text().split()
    return Func(lambda: words)
"""


@pytest.mark.parametrize(
    ("source", "refusal"),
    [
        (HELPED, "raised ModuleNotFoundError: No module named 'helper'"),
        (READS_A_FILE, "raised FileNotFoundError"),
    ],
    ids=["a-module", "a-file"],
)
def test_a_tool_needing_what_only_this_chats_folder_has_doesnt_build(
    fresh, tmp_path, monkeypatch, source, refusal
):
    # The worker runs in the chat's folder, with it last on sys.path.
    folder = tmp_path / "chat-folder"
    folder.mkdir()
    (folder / "helper.py").write_text("def count(text):\n    return len(text)\n")
    (folder / "words.txt").write_text("a b c")
    monkeypatch.chdir(folder)
    monkeypatch.syspath_prepend(str(folder))
    with pytest.raises(LibraryError, match=re.escape(refusal)) as refused:
        library_tool(fresh).write_tool("counter", source, "build")
    assert "doesn't build, so it wasn't written" in str(refused.value)
    assert stored_tools(fresh) == []


def stored_as(library, name, source):
    """Tool *name* of *source*, built by its build function, stored as an import stores
    one, granted nowhere."""
    block = dump({"factory": "build", "factory_from": f"{name}.py"})
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(UPSERT_TOOL, (name, block, source))


@pytest.mark.parametrize(
    ("method", "name"), [("add_namespace", "research"), ("change_namespace", "drafts")]
)
def test_granting_a_tool_that_doesnt_build_is_refused_as_it_would_stop_every_chat(
    fresh, method, name
):
    stored_as(fresh, "broken", RAISING)
    before = stored_namespaces(fresh)
    refusal = (
        "Tool broken doesn't build, so it wasn't granted (granted, it would stop every"
        " chat; fix it with write_tool first): build in broken.py raised RuntimeError:"
        " built"
    )
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        getattr(library_tool(fresh), method)(name, "tools: [broken]\n")
    assert stored_namespaces(fresh) == before


def test_granting_a_tool_that_builds_grants_it(fresh):
    stored_as(fresh, "counter", WORD_COUNT)
    library_tool(fresh).change_namespace("drafts", "tools: [counter]\n")
    assert own_fields(fresh, "drafts") == {"tools": ["counter"]}


def test_a_tool_changed_while_it_was_built_for_a_grant_is_not_granted(fresh):
    # Its build replaces it in the Library, as another chat's write would meanwhile.
    replaces_itself = f"""import sqlite3

from deep_reasoner.primitives import Func


def build(client, params):
    with sqlite3.connect({str(fresh)!r}) as connection:
        connection.execute("UPDATE tools SET source = source || '#' WHERE name = 'sly'")
    return Func(len)
"""
    stored_as(fresh, "sly", replaces_itself)
    before = stored_namespaces(fresh)
    with pytest.raises(LibraryError, match=re.escape(CHANGED_MEANWHILE)):
        library_tool(fresh).change_namespace("drafts", "tools: [sly]\n")
    assert stored_namespaces(fresh) == before


# A factory that leaves a mark when it runs.
MARKS = """from pathlib import Path

from deep_reasoner.primitives import Func


def build(client, params):
    Path({marker!r}).write_text("ran")
    return Func(len)
"""


@pytest.mark.parametrize(
    ("method", "name", "text", "refusal"),
    [
        (
            "add_namespace",
            "nowhere.child",
            "tools: [marks]\n",
            'Add namespace "nowhere" first',
        ),
        ("add_namespace", "drafts", "tools: [marks]\n", "There is already a namespace"),
        ("add_namespace", "re search", "tools: [marks]\n", "A namespace name is words"),
        (
            "add_namespace",
            "research",
            "tools: [marks]\nspawn: [nowhere]\n",
            'spawn names "nowhere"',
        ),
        (
            "add_namespace",
            "research",
            "tools: [marks]\nvars: {max: 3}\n",
            "would hide Python's own max",
        ),
        ("change_namespace", "nowhere", "tools: [marks]\n", "has no namespace"),
    ],
    ids=[
        "no-parent",
        "already-there",
        "a-bad-name",
        "spawn-to-none",
        "a-bad-var",
        "none",
    ],
)
def test_a_grant_refused_for_what_is_cheap_to_check_never_builds_the_tool(
    fresh, tmp_path, method, name, text, refusal
):
    marker = tmp_path / "ran.txt"
    stored_as(fresh, "marks", MARKS.format(marker=str(marker)))
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        getattr(library_tool(fresh), method)(name, text)
    assert not marker.exists()


def test_a_tool_another_namespace_grants_is_granted_without_building_it_again(
    fresh, tmp_path
):
    marker = tmp_path / "ran.txt"
    stored_as(fresh, "marks", MARKS.format(marker=str(marker)))
    with closing(open_library(fresh)) as connection:
        add_namespace(connection, "marking", "tools: [marks]\n")
    library_tool(fresh).change_namespace("drafts", "tools: [marks]\n")
    assert not marker.exists()
    assert own_fields(fresh, "drafts") == {"tools": ["marks"]}


def test_a_chat_whose_repl_became_a_sandbox_while_a_tool_was_built_may_not_grant_it(
    fresh,
):
    # Its build sandboxes the librarian, as the user might in the panel meanwhile.
    sandboxes = f"""import sqlite3

from deep_reasoner.primitives import Func


def build(client, params):
    with sqlite3.connect({str(fresh)!r}) as connection:
        connection.execute(
            "UPDATE namespaces SET yaml = 'tools: [library]\\nrepl: {{type: restricted}}\\n'"
            " WHERE name = 'librarian'"
        )
    return Func(len)
"""
    stored_as(fresh, "sandboxes", sandboxes)
    with pytest.raises(LibraryError, match=re.escape(NOT_LOCAL)):
        library_tool(fresh).change_namespace("drafts", "tools: [sandboxes]\n")
    assert ("drafts", "") in stored_namespaces(fresh)


def test_a_chat_whose_namespace_runs_a_sandboxed_repl_may_not_grant_a_tool_nor_build_it(
    fresh, tmp_path
):
    marker = tmp_path / "ran.txt"
    stored_as(fresh, "marks", MARKS.format(marker=str(marker)))
    sandboxed = "tools: [library]\nrepl: {type: restricted}\n"
    with closing(open_library(fresh)) as connection:
        change_namespace(connection, "librarian", sandboxed)
    with pytest.raises(LibraryError, match=re.escape(NOT_LOCAL)):
        library_tool(fresh).change_namespace("drafts", "tools: [marks]\n")
    assert not marker.exists()
    assert ("drafts", "") in stored_namespaces(fresh)


def test_tools_lists_every_tool_in_brief_and_gives_one_whole(fresh):
    tool = library_tool(fresh)
    long_source = WORD_COUNT + "#" * 500 + "\n"
    tool.write_tool("word_count", long_source, "build", {"sep": ","})
    tool.change_namespace("drafts", {"tools": ["word_count"]})
    by_name = {each["name"]: each for each in tool.tools()}
    assert by_name["word_count"] == {
        "name": "word_count",
        "kind": "custom",
        "factory": "build",
        "settings": {"sep": ","},
        "granted_in": ["drafts"],
        "description": "",
        "source": f"{long_source[:BRIEF_CHARS]}…(+{len(long_source) - BRIEF_CHARS}"
        " chars)",
    }
    assert (by_name["llm"]["kind"], by_name["llm"]["source"]) == ("built-in", None)
    assert (by_name["library"]["kind"], by_name["library"]["granted_in"]) == (
        "the app's",
        [LIBRARIAN],
    )
    assert tool.tools("word_count")["source"] == long_source
    assert json.loads(json.dumps(tool.tools())) == tool.tools()


def test_changes_list_this_chats_library_changes_in_its_messages(library):
    open_library(library).close()
    namespaces, tools = recorded("namespaces"), recorded("tools")
    plant(library, CHAT, namespaces)
    plant(library, ANOTHER_CHAT, recorded("librarian"))
    plant(library, CHAT, tools)
    changes = library_tool(library).changes()
    assert [(each["turn"], each["what"], each["name"]) for each in changes] == [
        (1, "namespace", "research"),
        (1, "namespace", "research"),
        (2, "tool", "word_count"),
        (2, "namespace", "counting"),
        (2, "tool", "word_count"),
    ]
    added, changed = changes[:2]
    assert added | {"after": None, "changed": None} == {
        "turn": 1,
        "message": "Set up research.",
        "what": "namespace",
        "namespace": "research",
        "name": "research",
        "before": None,
        "after": None,
        "changed": None,
    }
    assert (changed["before"], changed["after"]) == (added["after"], RESEARCH_CHANGED)
    assert changed["changed"] == {
        "system_suffix": {
            "before": "Cite a source for every claim.\n",
            "after": "Cite two sources for every claim.\n",
        }
    }
    assert changes[-1]["before"] == {
        "source": WORD_COUNT,
        "factory": "build",
        "settings": None,
    }
    assert changes[-1]["after"] == {
        "source": SPLIT_COUNT,
        "factory": "build",
        "settings": {"sep": ","},
    }
    assert library_tool(library).changes(2) == changes[2:]


def test_a_tools_before_written_back_restores_it_as_it_was(library):
    open_library(library).close()
    plant(library, CHAT, recorded("tools"))
    tool = library_tool(library)
    replaced = tool.changes()[-1]
    tool.write_tool("word_count", **replaced["before"])
    assert stored_tools(library) == [
        ("word_count", "factory: build\nfactory_from: word_count.py\n", WORD_COUNT)
    ]


def test_a_logged_tool_change_that_is_no_tools_row_reads_as_it_was_logged(library):
    open_library(library).close()
    forged = {"kind": "library.change", "what": "tool", "name": "x", "before": None}
    plant(library, CHAT, crafted(forged | {"after": "not a row"}))
    [change] = library_tool(library).changes()
    assert (change["before"], change["after"]) == (None, "not a row")


def test_changes_in_a_chats_first_message_list_its_own_so_far(fresh):
    assert library_tool(fresh).changes() == []
    # As this message's run saved it, when a cell has just written research.
    added = {"kind": "library.change", "what": "namespace", "name": "research"}
    added |= {"namespace": "research", "before": None, "after": "tools: [llm]\n"}
    plant(fresh, CHAT, crafted(added)[:-1])
    [change] = library_tool(fresh).changes()
    assert (change["turn"], change["changed"]) == (
        1,
        {"tools": {"before": None, "after": ["llm"]}},
    )


@pytest.mark.parametrize("turn", [0, 2, "1", True])
def test_changes_of_a_message_that_isnt_there_say_which_are(fresh, turn):
    plant(fresh, CHAT, crafted())
    refusal = f"There is no message {turn!r}: this chat's messages are numbered 1 to 1"
    with pytest.raises(ValueError, match=re.escape(refusal)):
        library_tool(fresh).changes(turn)


def test_a_logged_change_nested_too_deep_to_read_still_lists(library):
    open_library(library).close()
    deep = "[" * 50_000 + "]" * 50_000
    forged = {"kind": "library.change", "what": "namespace", "namespace": "x"}
    forged |= {"name": "x", "before": None, "after": deep}
    tool = {"kind": "library.change", "what": "tool", "name": "y", "before": None}
    plant(library, CHAT, crafted(forged, tool | {"after": f"{deep}\n# y.py\n"}))
    namespace, tool = library_tool(library).changes()
    assert (namespace["changed"], tool["after"]) == ({}, f"{deep}\n# y.py\n")


def test_a_logged_tool_without_a_factory_is_built_by_its_name(library):
    open_library(library).close()
    logged = {"kind": "library.change", "what": "tool", "name": "y", "before": None}
    plant(
        library,
        CHAT,
        crafted(logged | {"after": "factory_from: y.py\n# y.py\nx = 1\n"}),
    )
    [change] = library_tool(library).changes()
    assert change["after"] == {"source": "x = 1\n", "factory": "y", "settings": None}


def test_the_seeded_worked_example_written_back_stays_as_it_was(fresh):
    [text] = [each for each in LIBRARIANS_EXAMPLES if "[repl]" in each]
    tool = library_tool(fresh)
    done = tool.write_decomposition(LIBRARIAN, text)
    [row] = [
        each
        for each in tool.show(LIBRARIAN)["decompositions"]
        if each["name"] == done["name"]
    ]
    assert (done["done"], row["yaml"], row["problems"]) == ("replaced", text, [])


def test_a_merge_past_the_yaml_limit_is_refused_and_changes_nothing(fresh):
    half = "a" * 600_000
    tool = library_tool(fresh)
    tool.add_namespace("research", {"system_suffix": half})
    before = stored_namespaces(fresh)
    refusal = "A namespace's YAML may be at most 1,000,000 characters; with this change"
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        tool.change_namespace("research", {"vars": {"x": half}})
    assert stored_namespaces(fresh) == before


@pytest.mark.parametrize(
    "write",
    [
        lambda tool, cut_text: tool.change_namespace(
            "drafts", {"system_suffix": cut_text}
        ),
        lambda tool, cut_text: tool.write_decomposition(
            "drafts", EXAMPLE.replace("How many tickets are open?", cut_text)
        ),
        lambda tool, cut_text: tool.write_tool(
            "counter", f"{WORD_COUNT}# {cut_text}\n", "build"
        ),
    ],
    ids=["namespace", "decomposition", "tool"],
)
def test_what_a_brief_list_cut_short_is_never_written_back(fresh, write):
    tool = library_tool(fresh)
    tool.add_namespace("research", {"system_suffix": "Cite. " * 100})
    [research] = [each for each in tool.show() if each["name"] == "research"]
    cut_text = research["sets"]["system_suffix"]
    before = (stored(fresh), stored_namespaces(fresh), stored_tools(fresh))
    with pytest.raises(LibraryError, match=re.escape(CUT_SHORT)):
        write(tool, cut_text)
    assert (stored(fresh), stored_namespaces(fresh), stored_tools(fresh)) == before


def page_of(library, chat=CHAT, key_names=()):
    """The page tool, as the run CURRENT of *chat* gets it, in an app file the app has
    laid out."""
    open_library(library).close()
    told = {
        "library": str(library),
        "chat": chat,
        "run": CURRENT,
        "key_names": list(key_names),
    }
    return build_page(None, told).value


def saved(library, p):
    """The blocks page *p* saves when opened, as the reader reads them back."""
    p.open()
    [(page, *_), *_] = stored_pages(library)[::-1]
    with closing(open_library(library)) as connection:
        return read_page(connection, page)["blocks"]


def as_listed(step):
    """A page's copy of a step as history.steps lists it: a cell's sub-agents by node."""
    if step["kind"] != "cell":
        return step
    return step | {"agents": list(dict.fromkeys(row["node"] for row in step["agents"]))}


def drives_in(rows):
    """Each sub-agent drive in a page's copy, and each one below it."""
    stack, found = list(rows), []
    while stack:
        row = stack.pop()
        found.append(row)
        stack += [each for step in row["steps"] for each in step.get("agents", [])]
    return found


AGENT_FIELDS = ("node", "drive", "task", "namespace", "status", "result", "cost")
RECORDED = [
    "fanout",
    "nested",
    "drives",
    "failing",
    "raising",
    "stopped",
    "librarian",
    "pages",
]


def whole_steps(history, turn, node):
    """history.steps(turn, node), each step whole, as history.full gives it."""
    listed = history.steps(turn, node=node)
    return [history.full(turn, node, index) for index in range(len(listed))]


@pytest.mark.parametrize("name", RECORDED)
def test_a_step_on_a_page_is_the_step_history_lists_there_whole(library, name):
    plant(library, CHAT, recorded(name))
    history, page = history_of(library), page_of(library)
    nodes = [None, *dict.fromkeys(agent["node"] for agent in history.agents(1))]
    for node in nodes:
        listed = whole_steps(history, 1, node)
        p = page.new("Steps")
        for index in range(len(listed)):
            p.step(1, node, index=index)
        shown = saved(library, p) if listed else []
        assert [as_listed(block["step"]) for block in shown] == listed


@pytest.mark.parametrize("name", RECORDED)
def test_a_sub_agent_on_a_page_is_its_drives_history_lists_with_their_steps_below(
    library, name
):
    plant(library, CHAT, recorded(name))
    history, page = history_of(library), page_of(library)
    agents = history.agents(1)
    for node in dict.fromkeys(agent["node"] for agent in agents):
        p = page.new("A sub-agent")
        p.agent(1, node, note="Here it is.")
        [block] = saved(library, p)
        listed = [agent for agent in agents if agent["node"] == node]
        shown = [{key: row[key] for key in AGENT_FIELDS} for row in block["agents"]]
        assert shown == [{key: agent[key] for key in AGENT_FIELDS} for agent in listed]
        assert [len(row["steps"]) for row in block["agents"]] == [
            agent["steps"] for agent in listed
        ]
        for row in drives_in(block["agents"]):
            steps = whole_steps(history, 1, row["node"])
            of_drive = [step for step in steps if step["drive"] == row["drive"]]
            assert [as_listed(step) for step in row["steps"]] == of_drive


def test_a_delegating_step_carries_each_sub_agent_it_started_with_its_time(library):
    events = recorded("fanout")
    plant(library, CHAT, events)
    p = page_of(library).new("The fan-out")
    p.step(1, index=-2)
    [block] = saved(library, p)
    starts = {e["node"]: e["t"] for e in events if e["kind"] == "agent.start"}
    ends = {e["node"]: e["t"] for e in events if e["kind"] == "agent.end"}
    assert (block["index"], block["task"]) == (1, None)
    assert [
        (row["task"], row["seconds"], row["unpriced"])
        for row in block["step"]["agents"]
    ] == [
        ("Summarize C1.", ends[2] - starts[2], False),
        ("Summarize C2.", ends[3] - starts[3], False),
    ]


@pytest.mark.parametrize(
    ("add", "refusal"),
    [
        (lambda p: p.step(2, index=0), "There is no message 2: this chat's earlier"),
        (lambda p: p.step(True, index=0), "There is no message True"),
        (
            lambda p: p.step(1, node=9, index=0),
            "Message 1 has no agent 9: its agents' nodes are [1, 2, 3].",
        ),
        (
            lambda p: p.step(1, index=2),
            "Message 1's top agent has 2 steps, numbered from 0: there is no step 2.",
        ),
        (
            lambda p: p.step(1, node=3, index=-2),
            "Message 1's agent 3 has 1 step, numbered from 0: there is no step -2.",
        ),
        (lambda p: p.step(1, index="0"), NOT_AN_INDEX),
        (lambda p: p.step(1, index=False), NOT_AN_INDEX),
        (lambda p: p.step(1, node=2.0, index=0), NOT_A_REF),
        (lambda p: p.step(1, index=0, note=["Why"]), NOT_A_REF),
        (
            lambda p: p.agent(1, 9),
            "Message 1 has no agent 9: its agents' nodes are [1, 2, 3].",
        ),
        (lambda p: p.agent(1, 1), "Message 1's top agent is no sub-agent"),
        (lambda p: p.agent(1, None), "Message 1's top agent is no sub-agent"),
    ],
    ids=[
        "a-message-not-there",
        "a-bool-for-a-message",
        "a-node-not-there",
        "an-index-past-the-end",
        "an-index-before-the-start",
        "an-index-not-a-number",
        "a-bool-for-an-index",
        "a-node-not-an-int",
        "a-note-not-text",
        "a-sub-agent-not-there",
        "the-top-agent-as-a-sub-agent",
        "no-node",
    ],
)
def test_a_reference_to_what_isnt_there_is_refused_in_words_naming_what_is(
    library, add, refusal
):
    plant(library, CHAT, recorded("nested"))
    p = page_of(library).new("Refused")
    with pytest.raises(ValueError, match=re.escape(refusal)):
        add(p)
    with pytest.raises(ValueError, match=re.escape(EMPTY)):
        p.open()


def test_a_chat_with_no_earlier_messages_has_nothing_to_point_at(library):
    p = page_of(library).new("Nothing yet")
    with pytest.raises(ValueError, match="no earlier messages: this is its first"):
        p.step(1, index=0)


ROWS_OF_TEXT = [{"course": "CS101", "seats": 3}]


@pytest.mark.parametrize(
    ("add", "refusal"),
    [
        (lambda p: p.text(5), NOT_MARKDOWN),
        (lambda p: p.text(NeverEqual("x")), NOT_MARKDOWN),
        (
            lambda p: p.text("x" * 100_001),
            "A text may be at most 100,000 characters; this one is 100,001.",
        ),
        (
            lambda p: p.step(1, index=0, note="x" * 100_001),
            "A note may be at most 100,000 characters",
        ),
        (lambda p: p.table([{"n": i} for i in range(1_001)]), "at most 1,000 rows"),
        (lambda p: p.table([list(range(51))]), "at most 50 columns; this has 51"),
        (
            lambda p: p.table([{}], columns=[str(i) for i in range(51)]),
            "at most 50 columns",
        ),
        (lambda p: p.table(ROWS_OF_TEXT + [["CS201", 4]]), NOT_ROWS),
        (lambda p: p.table([{1: "CS101"}]), NOT_ROWS),
        (lambda p: p.table(ROWS_OF_TEXT, columns="course"), NOT_ROWS),
        (lambda p: p.table({"course": "CS101"}), NOT_ROWS),
        (lambda p: p.table([{"courses": ["CS101"]}]), NOT_VALUES),
        (lambda p: p.table([[("CS101",)]]), NOT_VALUES),
        (lambda p: p.table([HOLDS_ITSELF]), NOT_VALUES),
    ],
    ids=[
        "text-not-text",
        "text-not-plain-text",
        "a-long-text",
        "a-long-note",
        "too-many-rows",
        "too-many-columns",
        "too-many-columns-named",
        "rows-of-two-kinds",
        "a-key-not-text",
        "columns-not-a-list",
        "rows-not-a-list",
        "a-list-value",
        "a-tuple-value",
        "a-row-holding-itself",
    ],
)
def test_what_a_page_cant_hold_is_refused_in_words(library, add, refusal):
    plant(library, CHAT, recorded("nested"))
    p = page_of(library).new("Refused")
    with pytest.raises(ValueError, match=re.escape(refusal)):
        add(p)
    with pytest.raises(ValueError, match=re.escape(EMPTY)):
        p.open()


@pytest.mark.parametrize("title", ["", "  ", "x" * 201, 7, NeverEqual("Why")])
def test_a_title_must_be_text_of_1_to_200_characters(library, title):
    with pytest.raises(ValueError, match=re.escape(TITLE)):
        page_of(library).new(title)


def test_a_page_holds_at_most_200_blocks(library):
    p = page_of(library).new("Full")
    for _ in range(200):
        p.text("More.")
    with pytest.raises(ValueError, match="at most 200 blocks"):
        p.text("One too many.")
    assert len(saved(library, p)) == 200


def test_a_page_is_at_most_2_000_000_characters_as_saved(library):
    p = page_of(library).new("Long")
    with pytest.raises(ValueError, match="past 2,000,000: show less"):
        while True:
            p.text("x" * 100_000)
    assert len(saved(library, p)) == 19
    [(_, _, _, _, _, blocks)] = stored_pages(library)
    assert len(blocks) <= MAX_PAGE_CHARS


def stored_pages(library):
    with closing(open_library(library)) as connection:
        return connection.execute(
            "SELECT id, chat, run, node, title, blocks FROM pages"
        ).fetchall()


def test_a_page_is_saved_only_when_opened_and_reads_back_as_it_was_made(library):
    plant(library, CHAT, recorded("nested"))
    p = page_of(library).new("How the term was planned")
    p.text("It asked a sub-agent to **survey** CS.")
    p.step(1, index=0, note="The survey.")
    p.agent(1, 3)
    p.table([["CS101", 3], ["CS201", None]], columns=["course", "seats"])
    assert stored_pages(library) == []
    opened = p.open()
    [(page, chat, run, node, title, blocks)] = stored_pages(library)
    with closing(open_library(library)) as connection:
        read = read_page(connection, page)
    assert opened == (
        'Opened the page "How the term was planned"; the chat links it under your answer.'
    )
    assert page not in opened
    assert (chat, run, node, title) == (CHAT, CURRENT, None, "How the term was planned")
    assert read == {"id": page, "title": title, "blocks": json.loads(blocks)}
    assert [block["type"] for block in read["blocks"]] == [
        "text",
        "step",
        "agent",
        "table",
    ]
    assert read["blocks"][-1] == {
        "type": "table",
        "columns": ["course", "seats"],
        "rows": [["CS101", "3"], ["CS201", ""]],
    }


def test_a_table_of_dicts_takes_their_keys_as_columns_unless_given(library):
    rows = [{"course": "CS101", "seats": 3}, {"course": "CS201", "full": True}]
    p = page_of(library).new("Seats")
    p.table(rows)
    p.table(rows, columns=["full", "course"])
    assert [(block["columns"], block["rows"]) for block in saved(library, p)] == [
        (["course", "seats", "full"], [["CS101", "3", ""], ["CS201", "", "True"]]),
        (["full", "course"], [["", "CS101"], ["True", "CS201"]]),
    ]


def test_an_empty_page_is_not_opened(library):
    with pytest.raises(ValueError, match=re.escape(EMPTY)):
        page_of(library).new("Empty").open()
    assert stored_pages(library) == []


def test_a_page_opened_again_is_saved_as_it_stands_in_its_place(library):
    p = page_of(library).new("Twice")
    p.text("First.")
    first = p.open()
    p.text("Second.")
    second = p.open()
    [(page, *_)] = stored_pages(library)
    with closing(open_library(library)) as connection:
        read = read_page(connection, page)
    assert [block["text"] for block in read["blocks"]] == ["First.", "Second."]
    assert (first, second) == (
        'Opened the page "Twice"; the chat links it under your answer.',
        'Saved the page "Twice" again, as it stands now; the chat still links it.',
    )


def test_a_page_deleted_before_it_is_opened_again_is_a_refusal_in_words(library):
    p = page_of(library).new("Gone")
    p.text("First.")
    p.open()
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute("DELETE FROM pages")
    with pytest.raises(LibraryError, match="no longer in the Library file"):
        p.open()


def test_a_providers_key_is_withheld_from_all_a_page_holds(library, monkeypatch):
    monkeypatch.setenv(SHELL_KEY_ENV, SHELL_KEY)
    plant(library, CHAT, crafted(*ROOT_AND_CELL))
    with closing(open_library(library)) as connection, writing(connection):
        # As a file from another machine may hold: the key in a step of the record.
        connection.execute(
            "UPDATE events SET data = replace(data, 'kept', ?)", (SHELL_KEY,)
        )
    p = page_of(library, key_names=[SHELL_KEY_ENV]).new(f"Mine: {SHELL_KEY}")
    p.text(f"The key is {SHELL_KEY}.")
    p.step(1, index=0, note=SHELL_KEY)
    p.table([{SHELL_KEY: SHELL_KEY}])
    p.open()
    [(_, _, _, _, title, blocks)] = stored_pages(library)
    assert SHELL_KEY not in title + blocks
    assert title == f"Mine: {KEY_WITHHELD}"
    assert json.loads(blocks)[0]["text"] == f"The key is {KEY_WITHHELD}."
    assert json.loads(blocks)[1]["step"]["code"] == f"print('{KEY_WITHHELD}')"


def test_a_busy_app_file_is_a_refusal_to_open_in_words(library):
    p = page_of(library).new("Busy")
    p.text("Wait.")
    refused = pytest.raises(LibraryError, match="can't be used now .*; try again")
    with closing(open_library(library)) as connection, writing(connection), refused:
        p.open()
    assert stored_pages(library) == []


def test_a_moved_app_file_is_never_made_afresh_by_a_page(library, tmp_path):
    p = page_of(library).new("Moved")
    p.text("Gone.")
    library.rename(tmp_path / "moved.sqlite")
    with pytest.raises(LibraryError, match="can't be used now"):
        p.open()
    assert not library.exists()


def test_a_cell_setting_the_pages_state_itself_saves_no_key_and_nothing_unchecked(
    library, monkeypatch
):
    monkeypatch.setenv(SHELL_KEY_ENV, SHELL_KEY)
    p = page_of(library, key_names=[SHELL_KEY_ENV]).new("Mine")
    p.text("Checked.")
    p.title, p.blocks = SHELL_KEY, [{"type": "text", "text": SHELL_KEY}]
    p._title = f"Set by a cell: {SHELL_KEY}"
    p._blocks.append({"type": "text", "text": f"Set by a cell: {SHELL_KEY}"})
    blocks = saved(library, p)
    [(_, _, _, _, title, text)] = stored_pages(library)
    assert SHELL_KEY not in title + text
    assert title == f"Set by a cell: {KEY_WITHHELD}"
    assert blocks == [
        {"type": "text", "text": "Checked."},
        {"type": "text", "text": f"Set by a cell: {KEY_WITHHELD}"},
    ]


@pytest.mark.parametrize(
    ("tamper", "refusal"),
    [
        (lambda p: setattr(p, "_title", "x" * 201), TITLE),
        (lambda p: p._blocks.append({"type": "chart"}), NOT_SAVED),
        (
            lambda p: p._blocks.append({"type": "text", "text": "x" * 100_001}),
            NOT_SAVED,
        ),
        (lambda p: p._blocks.append({"type": "text", "text": ("x",)}), NOT_SAVED),
        (lambda p: p._blocks.append({"type": "text", "text": {1, 2}}), NOT_SAVED),
        (lambda p: p._blocks[0]["agents"][0].update(cost=2.0**60), NOT_SAVED),
        (lambda p: p._blocks.append(HOLDS_ITSELF), TOO_DEEP),
        (lambda p: setattr(p, "_blocks", "blocks"), NOT_SAVED),
    ],
    ids=[
        "a-long-title",
        "an-unknown-block",
        "a-long-text",
        "a-tuple",
        "a-set",
        "a-cost-past-2-to-the-53",
        "a-list-holding-itself",
        "blocks-not-a-list",
    ],
)
def test_what_a_cell_sets_in_the_pages_state_is_refused_at_open(
    library, tamper, refusal
):
    plant(library, CHAT, recorded("nested"))
    p = page_of(library).new("Tampered")
    p.agent(1, 2)
    tamper(p)
    with pytest.raises(ValueError, match=re.escape(refusal)):
        p.open()
    assert stored_pages(library) == []


ROOT_CELL_AND_SUB_AGENT = (
    *ROOT_AND_CELL,
    {"kind": "agent.start", "node": 2, "parent": 1, "cell": 1, "task": "Sum."},
)
OVER_RANGE = {
    "cost": [
        *ROOT_CELL_AND_SUB_AGENT,
        {"kind": "usage", "node": 2, "cost": 2**53},
        {"kind": "usage", "node": 2, "cost": 2**53},
    ],
    "seconds": [
        *ROOT_AND_CELL,
        {"kind": "agent.start", "node": 2, "parent": 1, "cell": 1, "task": "Sum."}
        | {"t": -(2**53)},
        {"kind": "agent.end", "node": 2, "status": "done", "t": 2**53},
    ],
}


@pytest.mark.parametrize("what", OVER_RANGE)
def test_a_number_a_crafted_record_sums_past_2_to_the_53_is_refused_not_saved_unread(
    library, what
):
    plant(library, CHAT, crafted(*OVER_RANGE[what]))
    p = page_of(library).new("Over")
    for add in (lambda: p.agent(1, 2), lambda: p.step(1, index=0)):
        with pytest.raises(ValueError, match=re.escape(NOT_SHOWN)):
            add()
    with pytest.raises(ValueError, match=re.escape(EMPTY)):
        p.open()


def nested_run(levels):
    """A run's events in which each sub-agent starts the next, *levels* deep, as a file
    from someone else may hold."""
    events = [{"kind": "agent.start", "node": 1, "parent": None, "cell": None}]
    for node in range(1, levels + 1):
        events.append({"kind": "cell.start", "node": node, "cell": node, "code": ""})
        events.append(
            {"kind": "agent.start", "node": node + 1, "parent": node, "cell": node}
            | {"task": "Deeper."}
        )
    return crafted(*events)


def test_a_record_nested_too_deep_to_copy_is_refused_in_words(library):
    plant(library, CHAT, nested_run(600))
    p = page_of(library).new("Deep")
    for add in (lambda: p.agent(1, 2), lambda: p.step(1, index=0)):
        with pytest.raises(ValueError, match=re.escape(TOO_DEEP)):
            add()
