import json
import re
import sqlite3
from contextlib import closing
from itertools import count
from pathlib import Path

import pytest
from fake_model import WORD_COUNT
from structlog.testing import capture_logs

from deep_reasoning.app_tools import (
    MAX_YAML_CHARS,
    NOT_A_TOOL,
    NOT_LOCAL,
    NOT_TEXT,
    SETS_OWN_KEYS,
    build_history,
    build_library,
    open_existing,
)
from deep_reasoning.library import (
    HISTORIAN,
    HISTORIANS_EXAMPLES,
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
from deep_reasoning.runs import CHANGE_FIELDS, LOST, chat_runs, read_turn, save_event

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


def library_tool(library, namespace="librarian"):
    """The library tool, as a run of a chat in *namespace* gets it."""
    told = {
        "library": str(library),
        "chat": CHAT,
        "run": CURRENT,
        "tools_dir": str(library.parent / "run-tools"),
        "namespace": namespace,
        "key_names": [SHELL_KEY_ENV],
    }
    return build_library(None, told).value


# A fresh app file's decompositions: root's hand-offs, then the historian's examples of the
# same names.
FRESH = [
    (namespace, parsed(text).name)
    for namespace in ("root", HISTORIAN)
    for text in HISTORIANS_EXAMPLES
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


def test_show_gives_every_namespace_with_its_decompositions_as_plain_data(fresh):
    tool = library_tool(fresh)
    done = tool.write_decomposition("librarian", EXAMPLE)
    shown = tool.show()
    by_name = {each["name"]: each for each in shown}
    assert done == {
        "done": "added",
        "what": "decomposition",
        "namespace": "librarian",
        "name": "count the tickets",
        "problems": ONLY_THE_TASK,
    }
    assert json.loads(json.dumps(shown)) == shown
    assert list(by_name) == ["root", "drafts", HISTORIAN, "librarian"]
    assert by_name["librarian"]["yaml"] == "tools:\n- library\n"
    assert by_name["librarian"]["decompositions"] == [
        {"name": "count the tickets", "yaml": EXAMPLE, "problems": ONLY_THE_TASK}
    ]


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
    by_name = {each["name"]: each for each in tool.show()}
    assert by_name["drafts"]["decompositions"] == [
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
    ],
    ids=[
        "text-unequal-to-root",
        "yaml-not-plain-text",
        "root",
        "too-long",
        "a-key",
        "an-escaped-key",
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
    [(change_namespace, "tools: [llm]\n"), (delete_namespace, 0)],
)
def test_a_chat_whose_namespace_lost_the_grant_mid_run_may_not_write(
    fresh, change, argument
):
    tool = library_tool(fresh)
    with closing(open_library(fresh)) as connection:
        change(connection, "librarian", argument)
    with pytest.raises(LibraryError, match=NOT_THIS_CHAT):
        tool.write_decomposition("drafts", EXAMPLE)
    assert stored(fresh) == FRESH


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
    }
    assert ("drafts", changed) in stored_namespaces(fresh)


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
    assert ("drafts", changed) in stored_namespaces(coder)


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
    refusal = "coder is built by the engine's own claude_code factory, which write_tool"
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
        "root grants search: replacing it is the user's choice, in the Library panel."
    )
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        tool.write_tool("search", RAISING, "build")
    assert stored_tools(fresh) == before


def test_a_tool_is_built_from_a_file_deleted_after_the_build(fresh):
    tool = library_tool(fresh)
    tool.write_tool("counter", WORD_COUNT, "build")
    with pytest.raises(LibraryError, match="raised RuntimeError: built"):
        tool.write_tool("other", RAISING, "build")
    assert list((fresh.parent / "run-tools" / "checks").iterdir()) == []
