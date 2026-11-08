import json
import re
import socket
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date
from pathlib import Path

import deep_reasoner.llm
import pytest
import yaml
from deep_reasoner.config import resolve_api_key
from deep_reasoner.namespaces import ROOT, build_registry
from deep_reasoner.tools import load_tool_factory
from deep_reasoner.v2 import DeepReasoner
from deep_reasoner.v2.agent import _render_context, _rendered_turns
from deep_reasoner.v2.cli import (
    V2Config,
    build_namespace_registry,
    build_reasoner,
    close_run,
    declared_tool_names,
)
from deep_reasoner.v2.messages import func, var
from structlog.testing import capture_logs

from deep_reasoning.importing import import_config
from deep_reasoning.library import (
    APP_TOOLS_FILE,
    BEFORE_EVENTS,
    BEFORE_PAGES,
    FILLED_IN,
    HAND_OFF_TURNS,
    HISTORIAN,
    HISTORIAN_SUFFIX,
    HISTORIANS_EXAMPLES,
    HISTORY_DESCRIPTION,
    LIBRARIAN,
    LIBRARIAN_SUFFIX,
    LIBRARIANS_EXAMPLES,
    LIBRARY_DESCRIPTION,
    LIBRARY_ONLY,
    NO_DESCRIPTION,
    PAGE_DESCRIPTION,
    UNREAD_SETTING,
    UPSERT_DECOMPOSITION,
    UPSERT_NAMESPACE,
    UPSERT_SETTING,
    UPSERT_TOOL,
    ChangedMeanwhile,
    LibraryError,
    add_decomposition,
    add_namespace,
    change_namespace,
    default_namespace,
    delete_decomposition,
    delete_namespace,
    delete_setting,
    delete_tool,
    dump,
    edit_decomposition,
    engine_config,
    hand_off,
    handed_to,
    open_library,
    panel,
    parsed,
    running_code,
    sandbox_templates,
    store_tool,
    stored_config,
    turn_problems,
    writing,
)

FAKE_KEY = "sk-test-not-a-real-key"
CHAT = "a-chat"
RUN = "20261027-120000-000000"
# advisors.yaml's namespaces, imported into a fresh Library beside its historian and librarian.
IMPORTED_NAMESPACES = {
    "root",
    "router",
    "course_advisor",
    "health_advisor",
    HISTORIAN,
    LIBRARIAN,
}
ROUTER_DECOMPOSITIONS = [
    "route a course question",
    "route a wellbeing question",
    "decline anything else",
]
TYPED = """system_prompt: |
  Solve it.
namespaces:
  root:
    vars: {by_year: {2024: 10}, due: 2026-10-25, cap: .inf}
    decompositions:
      - name: count
        messages:
          - {role: user, content: "How many?"}
          - role: assistant
            content: |
              <repl>
              FinalAnswer(3)
              </repl>
"""
ADDED = "name: added\nmessages: [{role: user, content: hi}]\n"
RESEARCH = "vars: {topic: ml}\nsystem_suffix: |\n  Cite each source.\n"
WAIT_FOR_THE_LOCK_S = 0.3
PANEL_READ_S = 5
# A build stopped at 5 seconds, with time for its process to start and be stopped.
WAIT_FOR_A_BUILD_S = 15
RAISES_ON_BUILD = "def build(client, params):\n    raise RuntimeError('offline')\n"
CONFIGS = Path(__file__).parent / "configs"
# Each case's turns and the problems they have, shared with tests/turns.test.mjs.
TURN_CASES = json.loads(
    (Path(__file__).parent / "turn_cases.json").read_text(encoding="utf-8")
)
NO_THINK = """name: no think
messages:
- {role: user, content: "How many?"}
- {role: assistant, content: "<repl>FinalAnswer(3)</repl>"}
"""
# A fresh Library's worked examples: the historian's, and root's hand-offs of the same names.
HISTORIANS = [parsed(text) for text in HISTORIANS_EXAMPLES]
HAND_OFFS = [parsed(hand_off(text)) for text in HISTORIANS_EXAMPLES]
LIBRARIANS = [parsed(text) for text in LIBRARIANS_EXAMPLES]
SEEDED_ROWS = [(each.name, []) for each in HAND_OFFS]
ROUTER_SHOWN = ROUTER_DECOMPOSITIONS + [each.name for each in HAND_OFFS]
# Root spawns a sub-agent into historian, with the user's message and this conversation.
HAND_OFF = 'FinalAnswer(subagent(ctx.task, namespace="historian", history=ctx.history))'
# What the app puts in for a chat on OpenAI's gpt-6-luna.
CHATS_MODEL = {
    "model": "gpt-6-luna",
    "client": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


@pytest.fixture
def no_network(monkeypatch):
    def refuse(sock, address):
        raise AssertionError(f"tried to connect to {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def run_config(library, tmp_path, namespace=ROOT):
    """The config of a run on gpt-6-luna, as the app makes it."""
    return V2Config.model_validate(
        engine_config(library, namespace, tmp_path / "tools", CHAT, RUN) | CHATS_MODEL
    )


def import_into(library, config):
    """What importing *config* into *library* changed, as the lines it says."""
    with closing(open_library(library)) as connection:
        return import_config(connection, config)["changes"]


def dumped(library):
    with closing(open_library(library)) as connection:
        return list(connection.iterdump())


def made_with_an_empty_root(library):
    """The Library as a version made it before a fresh root granted tools or held the
    worked example."""
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(UPSERT_NAMESPACE, (ROOT, dump({})))
        connection.execute("DELETE FROM decompositions WHERE namespace = ?", (ROOT,))


def test_a_fresh_library_has_root_and_leaves_the_model_and_its_key_to_the_chat(
    library, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    config = engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)
    assert list(config["namespaces"]) == [ROOT, HISTORIAN, LIBRARIAN]
    assert default_namespace(library) == ROOT
    assert "model" not in config and "client" not in config
    assert resolve_api_key(run_config(library, tmp_path).client) == FAKE_KEY
    assert FAKE_KEY.encode() not in library.read_bytes()


def test_a_fresh_library_grants_llm_at_root_history_and_page_to_the_historian_and_library_to_the_librarian(
    library, tmp_path
):
    config = run_config(library, tmp_path)
    build_namespace_registry(config).check(declared_tool_names(config))
    assert config.namespaces[ROOT].tools == ["llm"]
    assert config.namespaces[HISTORIAN].tools == ["history", "page"]
    assert config.namespaces[HISTORIAN].spawn == []
    assert config.namespaces[HISTORIAN].system_suffix == HISTORIAN_SUFFIX
    assert config.namespaces[LIBRARIAN].tools == ["library"]
    # It may ask the historian about earlier messages, but never spawn itself.
    assert config.namespaces[LIBRARIAN].spawn == [HISTORIAN]
    assert config.namespaces[LIBRARIAN].system_suffix == LIBRARIAN_SUFFIX


def test_every_namespace_but_the_historian_hands_it_questions_about_earlier_messages(
    library, tmp_path
):
    registry = registry_of(library, tmp_path)
    shown = {name: registry.resolve(name).decompositions for name in registry.names}
    assert shown == {
        ROOT: HAND_OFFS,
        HISTORIAN: HISTORIANS,
        LIBRARIAN: [*HAND_OFFS, *LIBRARIANS],
    }
    assert [each.name for each in HAND_OFFS] == [
        "explain an earlier answer",
        "tell a typed answer from a computed one",
        "say what each sub-agent did",
        "show it on a page",
        "put a table on a page",
        "find where it really went wrong",
        "say when there is nothing to show",
    ]
    for example in HAND_OFFS:
        assert example.messages[-1].content.endswith(f"<repl>\n{HAND_OFF}\n</repl>\n")
        assert "as written (ctx.task, never reworded)" in example.messages[-1].content


def last_turns(library, tmp_path):
    """Each namespace's worked examples as a run of it sees them: for each, its last
    turn."""
    registry = registry_of(library, tmp_path)
    return {
        name: {
            each.name: each.messages[-1].content
            for each in registry.resolve(name).decompositions
        }
        for name in registry.names
    }


def shown_hand_offs(library, tmp_path):
    """last_turns(), each as the code the engine runs from it."""
    return {
        name: {example: running_code(turn) for example, turn in turns.items()}
        for name, turns in last_turns(library, tmp_path).items()
    }


def panel_rows(library):
    return {row["name"]: row for row in panel_of(library)["namespaces"]}


def hand_offs_field(row):
    found = [field["items"] for field in row["fields"] if field["label"] == "Hand-offs"]
    return found[0] if found else None


CANT = "historian namespace, and this namespace's spawn list leaves it out"


def test_a_namespace_that_may_not_spawn_into_the_historian_says_it_cant_in_place_of_handing_off(
    library, tmp_path
):
    with closing(open_library(library)) as connection:
        add_namespace(connection, "trips", "spawn: [trips]\n")
        add_namespace(connection, "trips.hotels", "")
        add_namespace(connection, "trips.anywhere", "spawn: null\n")
        add_namespace(connection, "trips.with_history", "spawn: [historian]\n")
    shown = shown_hand_offs(library, tmp_path)
    # The librarian's spawn list names the historian, so it keeps the hand-offs as they are.
    for name in (ROOT, LIBRARIAN, "trips.with_history"):
        assert {shown[name][each.name] for each in HAND_OFFS} == {HAND_OFF}, name
    for name in ("trips", "trips.hotels"):
        assert list(shown[name]) == [each.name for each in HAND_OFFS]
        for code in shown[name].values():
            assert "subagent" not in code and CANT in code
    # spawn: null sets nothing, so trips.anywhere inherits trips's spawn list.
    assert shown["trips.anywhere"] == shown["trips"]
    # What is stored is left as it is, and the panel says what trips can't hand off.
    rows = panel_rows(library)
    assert rows["trips"]["decompositions"] == []
    assert hand_offs_field(rows["trips"]) == [
        {
            "value": "none to historian: this namespace's spawn list leaves it out. Its"
            f" agents say they can't answer what {len(HAND_OFFS)} worked examples would"
            " hand there.",
            "source": "set here",
            "here": True,
        }
    ]
    assert hand_offs_field(rows["trips.hotels"])[0]["source"] == "inherited from trips"
    for name in (ROOT, LIBRARIAN, "trips.with_history"):
        assert hand_offs_field(rows[name]) is None


def test_a_historian_with_no_examples_of_its_own_reads_the_record_in_place_of_handing_off(
    library, tmp_path
):
    # As a Library whose historian was added without them may be: its spawn: [] refuses
    # a spawn into itself too, and it is where the questions go.
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(
            "DELETE FROM decompositions WHERE namespace = ?", (HISTORIAN,)
        )
    turns = last_turns(library, tmp_path)[HISTORIAN]
    assert list(turns) == [each.name for each in HAND_OFFS]
    for turn in turns.values():
        assert running_code(turn) == "print(history.turns())"
        assert "I answer it here and never hand it on" in turn
    [item] = hand_offs_field(panel_rows(library)[HISTORIAN])
    instead = f"Its agents answer themselves what {len(HAND_OFFS)} worked examples"
    assert instead in item["value"]


def test_with_the_historian_deleted_no_namespace_hands_off_to_it(library, tmp_path):
    with closing(open_library(library)) as connection:
        # The librarian's spawn list names the historian: it goes first.
        with pytest.raises(LibraryError, match="First take historian out of the spawn"):
            delete_namespace(connection, HISTORIAN, len(HISTORIANS))
        change_namespace(connection, LIBRARIAN, "tools: [library]\n")
        delete_namespace(connection, HISTORIAN, len(HISTORIANS))
    shown = shown_hand_offs(library, tmp_path)
    for name in (ROOT, LIBRARIAN):
        hand_offs = {each.name: shown[name][each.name] for each in HAND_OFFS}
        assert list(hand_offs) == [each.name for each in HAND_OFFS]
        for code in hand_offs.values():
            assert "subagent" not in code
            assert "there is no historian namespace" in code
    [item] = hand_offs_field(panel_rows(library)[ROOT])
    assert item["value"].startswith(
        "none to historian: there is no historian namespace"
    )
    assert item["source"] == "not in the Library"


def ending_in(name, code):
    """A worked example whose one reply runs *code*."""
    reply = f"<think>Hand it on.</think>\n<repl>\n{code}\n</repl>\n"
    return dump(
        {
            "name": name,
            "messages": [
                {"role": "user", "content": "Do it."},
                {"role": "assistant", "content": reply},
            ],
        }
    )


SWITCHING = 'switch_namespace("law", ctx.task)'
PARTIAL = "r = subagent('Solve x.', namespace='law')\nFinalAnswer(r + ' done')"
# Each ending, as a worked example's last cell, and the namespace it hands the whole
# question to: one that delegates a part, or names no namespace a name could be, hands
# off nothing.
ENDINGS = [
    ('FinalAnswer(subagent(ctx.task, namespace="law"))', "law"),
    (SWITCHING, "law"),
    ('switch_namespace(namespace="law")', "law"),
    (
        (
            "x = [subagent('a', namespace='math')]\n"
            "FinalAnswer(subagent(ctx.task, namespace='law'))"
        ),
        "law",
    ),
    (PARTIAL, None),
    ("FinalAnswer(len(subagent('Solve x.', namespace='law')))", None),
    ("FinalAnswer(subagent(ctx.task))", None),
    ('FinalAnswer(subagent(ctx.task, namespace="no such {{ place }}"))', None),
    ("FinalAnswer(subagent(ctx.task, namespace=where))", None),
]


@pytest.mark.parametrize(("code", "target"), ENDINGS)
def test_a_worked_example_hands_off_only_the_whole_question_to_a_named_namespace(
    code, target
):
    assert handed_to(parsed(ending_in("ending", code))) == target


def test_a_namespace_that_may_not_switch_to_another_answers_itself_and_a_partial_delegation_stays(
    library, tmp_path
):
    with closing(open_library(library)) as connection:
        add_namespace(connection, "law", "")
        add_namespace(connection, "trips", "spawn: []\n")
        add_decomposition(connection, ROOT, ending_in("switching", SWITCHING))
        add_decomposition(connection, ROOT, ending_in("partial", PARTIAL))
    turns = last_turns(library, tmp_path)
    assert running_code(turns[ROOT]["switching"]) == SWITCHING
    # law doesn't read the record, so trips answers what it can itself.
    switching = turns["trips"]["switching"]
    assert running_code(switching) == "print(ctx.task)"
    assert "I answer it myself if I can, and say so if I can't" in switching
    assert turns["trips"]["partial"] == turns[ROOT]["partial"]


def test_a_run_config_holds_an_example_only_where_what_a_namespace_inherits_differs(
    library, tmp_path
):
    with closing(open_library(library)) as connection:
        add_namespace(connection, "trips", "spawn: [trips]\n")
        add_namespace(connection, "trips.hotels", "")
        add_namespace(connection, "trips.with_history", "spawn: [historian]\n")
    namespaces = engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)[
        "namespaces"
    ]
    held = {
        name: [each["name"] for each in fields["decompositions"]]
        for name, fields in namespaces.items()
    }
    hand_offs = [each.name for each in HAND_OFFS]
    # root holds its own; trips the replacements, which trips.hotels inherits as they
    # are; trips.with_history, which can hand off, the originals back.
    assert held == {
        ROOT: hand_offs,
        HISTORIAN: [each.name for each in HISTORIANS],
        LIBRARIAN: [each.name for each in LIBRARIANS],
        "trips": hand_offs,
        "trips.hotels": [],
        "trips.with_history": hand_offs,
    }
    assert namespaces["trips.with_history"]["decompositions"] == [
        each.model_dump() for each in HAND_OFFS
    ]


def test_deleting_a_namespace_warns_of_the_hand_offs_to_it_that_could_be_made(library):
    with closing(open_library(library)) as connection:
        add_namespace(connection, "trips", "spawn: [trips]\n")
        add_namespace(connection, "historian.deep", "")
        for namespace in ("trips", "historian.deep", LIBRARIAN):
            text = ending_in(f"from {namespace}", HAND_OFF)
            add_decomposition(connection, namespace, text)
    # root's, and the librarian's own; not trips's, which can't, or one under historian.
    assert panel_rows(library)[HISTORIAN]["handed_to"] == len(HAND_OFFS) + 1


def test_root_is_told_to_pass_the_message_as_written_only_in_its_examples(
    library, tmp_path
):
    # In root's suffix, the historian would inherit it, and hand off itself.
    namespaces = run_config(library, tmp_path).namespaces
    assert namespaces[ROOT].system_suffix is None
    assert "ctx.task" not in namespaces[HISTORIAN].system_suffix


@pytest.mark.parametrize("example", HISTORIANS, ids=lambda example: example.name)
def test_the_historian_reads_history_then_answers_in_a_later_cell(example):
    handing_off, reading, *working, answering = [
        turn.content for turn in example.messages if turn.role == "assistant"
    ]
    assert HAND_OFF in handing_off
    # After root's hand-off, the historian's task is the user's message.
    assert example.messages[HAND_OFF_TURNS] == example.messages[0]
    # It reads first, then reads more or builds a page, answering in none of those cells.
    assert "history." in reading
    assert all("history." in each or "page." in each for each in working)
    assert all("FinalAnswer" not in each for each in [reading, *working])
    assert "FinalAnswer" in answering and "history." not in answering
    # The answer says what was found, and leaves the page's link to the chat.
    assert "/pages/" not in answering and "linked below" not in answering


def test_a_library_made_with_an_empty_root_is_given_a_fresh_root_once(
    library, tmp_path
):
    made_with_an_empty_root(library)
    upgraded = dumped(library)
    assert dumped(library) == upgraded
    assert run_config(library, tmp_path).namespaces[ROOT].tools == ["llm"]


def test_a_read_only_library_with_an_empty_root_still_opens_as_it_is(
    library, monkeypatch
):
    made_with_an_empty_root(library)
    connect = sqlite3.connect

    def read_only(path, **options):
        return connect(f"{Path(path).as_uri()}?mode=ro", uri=True, **options)

    monkeypatch.setattr(sqlite3, "connect", read_only)
    with capture_logs() as logs, closing(open_library(library)) as connection:
        assert stored_config(connection)["namespaces"][ROOT] == {"decompositions": []}
    assert [entry["event"] for entry in logs] == ["library.root_not_upgraded"]


def test_a_library_that_still_holds_a_model_and_client_opens_and_they_go_unread(
    library, tmp_path
):
    old = {"model": "gpt-4o", "client": {"base_url": "https://example.com/v1"}}
    with closing(open_library(library)) as connection, writing(connection):
        connection.executemany(
            UPSERT_SETTING, [(key, dump(value)) for key, value in old.items()]
        )
    with closing(open_library(library)) as connection:
        shown = panel(connection)
    config = engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)
    assert "model" not in config and "client" not in config
    assert [row["label"] for row in shown["run_settings"]][:2] == [
        "max_iter",
        "max_depth",
    ]
    assert not {"model", "client"} & {row["label"] for row in shown["run_settings"]}


def test_a_fresh_library_carries_a_generic_repl_prompt(library, tmp_path):
    prompt = run_config(library, tmp_path).system_prompt
    assert "<repl>" in prompt
    assert "FinalAnswer" in prompt


@pytest.mark.parametrize("version", [0, LIBRARY_ONLY, BEFORE_EVENTS, BEFORE_PAGES])
def test_a_file_that_is_not_a_library_is_refused_and_left_alone(tmp_path, version):
    other = tmp_path / "chats.sqlite"
    with closing(sqlite3.connect(other)) as connection:
        connection.execute("CREATE TABLE chats (id TEXT)")
        connection.execute(f"PRAGMA user_version = {version}")
    before = other.read_bytes()
    with pytest.raises(LibraryError, match="is not a Library"):
        open_library(other)
    assert other.read_bytes() == before


@pytest.mark.parametrize("version", [LIBRARY_ONLY, BEFORE_EVENTS, BEFORE_PAGES])
def test_an_older_file_that_already_holds_a_later_versions_tables_is_refused_and_left_alone(
    library, version
):
    open_library(library).close()
    with closing(sqlite3.connect(library)) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
    before = library.read_bytes()
    with pytest.raises(LibraryError, match="is not a Library"):
        open_library(library)
    assert library.read_bytes() == before


def test_an_open_reader_does_not_hold_up_a_write(library):
    with closing(open_library(library)) as reader:
        stored_config(reader)
        with closing(open_library(library)) as writer:
            add_decomposition(writer, LIBRARIAN, ADDED)
        assert [
            each["name"]
            for each in stored_config(reader)["namespaces"][LIBRARIAN]["decompositions"]
        ] == [*(each.name for each in LIBRARIANS), "added"]


@pytest.mark.parametrize(
    "text",
    [
        "a\x85b",
        "line\n\x85next\n",
        "a\u2028b",
        "a\u2029b",
        "ends in a newline\n",
        "  padded  ",
        "é中🚀",
        "#",
        ": ",
        'it\'s "quoted"',
    ],
)
def test_dumped_text_reads_back_exactly(text):
    value = {"vars": {"note": text}}
    assert yaml.safe_load(dump(value)) == value


def test_dumped_non_ascii_text_stays_readable():
    assert "é中🚀" in dump({"note": "é中🚀"})


def test_a_custom_tools_source_round_trips_and_the_engine_loads_its_factory(
    library, waitlist, tmp_path
):
    import_into(library, waitlist)
    block = engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)["tools"][
        "waitlist"
    ]
    written = block["factory_from"]
    assert Path(written).read_text() == (waitlist.parent / "tools.py").read_text()
    build = load_tool_factory("waitlist", block["factory"], written, None)
    queue = build(None, {"capacity": 1}).value
    assert queue.join("ada", "CS201") == "enrolled"
    assert queue.join("grace", "CS201") == "waitlisted at 1"


def test_a_shared_librarys_tool_cannot_be_written_outside_the_tools_folder(
    library, tmp_path
):
    with closing(open_library(library)) as connection:
        connection.execute(
            "INSERT INTO tools VALUES (?, ?, ?)",
            ("../escaped", "factory: build\n", "def build(client, cfg): ...\n"),
        )
        connection.execute(UPSERT_NAMESPACE, (ROOT, dump({"tools": ["../escaped"]})))
    with pytest.raises(LibraryError, match="needs a Python name"):
        engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)
    assert not (tmp_path / "escaped.py").exists()


def test_a_runs_config_holds_only_the_tools_a_namespace_grants_and_the_panel_lists_all(
    library, waitlist, tmp_path
):
    import_into(library, waitlist)
    broken = ("unused", dump({"factory": "build", "factory_from": "unused.py"}))
    with closing(open_library(library)) as connection:
        connection.execute(UPSERT_TOOL, (*broken, "def build(:\n"))
        listed = [tool["name"] for tool in panel(connection)["tools"]]
    config = engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)
    assert list(config["tools"]) == ["waitlist", "history", "library", "page"]
    assert not (tmp_path / "tools" / "unused.py").exists()
    assert listed == ["llm", "waitlist", "unused", "history", "library", "page"]


def test_an_app_tool_is_in_a_runs_config_only_while_a_namespace_grants_it(
    library, tmp_path
):
    def tools():
        return list(
            engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)["tools"]
        )

    open_library(library).close()
    granted = tools()
    with closing(open_library(library)) as connection:
        change_namespace(connection, HISTORIAN, "tools: [history]\nspawn: []\n")
    assert (granted, tools()) == (
        ["history", "library", "page"],
        ["history", "library"],
    )


def test_a_tool_stored_over_another_gives_back_the_row_it_replaced(library):
    block = {"factory": "build", "factory_from": "mine.py"}
    first, second = "def build(client, params): 1\n", "def build(client, params): 2\n"
    with closing(open_library(library)) as connection:
        added = store_tool(connection, "mine", block, first)
        replaced = store_tool(connection, "mine", block | {"sep": ","}, second)
    assert added is None
    assert tuple(replaced) == (dump(block), first)


def test_the_panel_lists_a_tool_whose_source_is_too_deep_to_parse(library):
    deep = f"x = {'1+' * 300_000}1\n"
    block = dump({"factory": "build", "factory_from": "deep.py"})
    with closing(open_library(library)) as connection:
        connection.execute(UPSERT_TOOL, ("deep", block, deep))
        tools = {tool["name"]: tool for tool in panel(connection)["tools"]}
    assert tools["deep"]["description"] == NO_DESCRIPTION


COUNT_WORDS = """from deep_reasoner.primitives import Func

DESCRIPTION = "count_words(text) -> how many words text holds"


class Counter:
    \"\"\"Counts what it is shown.\"\"\"


def build(client, params):
    \"\"\"Build the word counter.\"\"\"

    def count_words(text):
        \"\"\"Count the words in text.\"\"\"
        return len(text.split())

    return {made}
"""
SHADOWED = """from deep_reasoner.primitives import Func


def count_words(text):
    \"\"\"Count the words in text.\"\"\"


class Helper:
    def build(self):
        return Func(count_words, description="a method's")


def build(client, params):
    def helper():
        def count_words(text):
            \"\"\"A nested def's.\"\"\"

    return Func(count_words)
"""


@pytest.mark.parametrize(
    ("source", "description"),
    [
        (
            COUNT_WORDS.format(made='Func(count_words, description="Counts.")'),
            "Counts.",
        ),
        (COUNT_WORDS.format(made='Func(count_words, "Counts.")'), "Counts."),
        (
            COUNT_WORDS.format(made='Func(value=count_words, description="Counts.")'),
            "Counts.",
        ),
        (
            COUNT_WORDS.format(made="Func(count_words, description=DESCRIPTION)"),
            "count_words(text) -> how many words text holds",
        ),
        (
            COUNT_WORDS.format(
                made='Func(count_words, description=params["description"])'
            ),
            "set when it's built: params['description']",
        ),
        (COUNT_WORDS.format(made="Func(count_words)"), "Count the words in text."),
        (
            COUNT_WORDS.format(made="Func(Counter(), description=None)"),
            "Counts what it is shown.",
        ),
        (COUNT_WORDS.format(made="Func(lambda text: 0)"), NO_DESCRIPTION),
        (COUNT_WORDS.format(made='Func(params["made"])'), NO_DESCRIPTION),
        (SHADOWED, "Count the words in text."),
        ("def build(client, params):\n    return 1\n", NO_DESCRIPTION),
        ("def build(client, params) 1\n", NO_DESCRIPTION),
    ],
    ids=[
        "keyword",
        "positional",
        "keyword-only",
        "constant",
        "set-when-built",
        "function-docstring",
        "class-docstring",
        "lambda",
        "unknown-value",
        "shadowed",
        "no-func",
        "not-python",
    ],
)
def test_a_custom_tools_description_is_what_its_source_tells_the_agent(
    library, source, description
):
    block = dump({"factory": "build", "factory_from": "count_words.py"})
    with closing(open_library(library)) as connection:
        connection.execute(UPSERT_TOOL, ("count_words", block, source))
        tools = {tool["name"]: tool for tool in panel(connection)["tools"]}
    assert tools["count_words"]["description"] == description


@pytest.mark.parametrize(
    ("tool", "factory"),
    [
        ("history", "build_history"),
        ("library", "build_library"),
        ("page", "build_page"),
    ],
)
def test_a_runs_config_tells_each_app_tool_the_app_file_the_chat_and_the_run(
    library, tmp_path, monkeypatch, tool, factory
):
    monkeypatch.chdir(library.parent)
    with closing(open_library(library)) as connection:
        add_namespace(connection, "granting", f"tools: [{tool}]\n")
    config = engine_config(Path(library.name), ROOT, tmp_path / "tools", CHAT, RUN)
    assert config["tools"][tool] == {
        "factory_from": str(APP_TOOLS_FILE),
        "factory": factory,
        "library": str(library.resolve()),
        "chat": CHAT,
        "run": RUN,
        "tools_dir": str((tmp_path / "tools").resolve()),
        "namespace": ROOT,
        "key_names": [],
    }
    with closing(open_library(library)) as connection:
        assert connection.execute("SELECT count(*) FROM tools").fetchone() == (0,)


@pytest.mark.parametrize(
    ("tool", "description"),
    [
        ("history", HISTORY_DESCRIPTION),
        ("library", LIBRARY_DESCRIPTION),
        ("page", PAGE_DESCRIPTION),
    ],
)
def test_a_shared_files_tool_named_as_an_app_tool_is_shadowed_by_the_apps(
    library, tmp_path, tool, description
):
    with closing(open_library(library)) as connection:
        add_namespace(connection, "granting", f"tools: [{tool}]\n")
        connection.execute(
            "INSERT INTO tools VALUES (?, ?, ?)",
            (tool, f"factory: build\nfactory_from: {tool}.py\n", "def build(): 1\n"),
        )
    with capture_logs() as logs:
        config = engine_config(library, ROOT, tmp_path / "tools", CHAT, RUN)
        with closing(open_library(library)) as connection:
            tools = {each["name"]: each for each in panel(connection)["tools"]}
    assert config["tools"][tool]["factory_from"] == str(APP_TOOLS_FILE)
    assert not (tmp_path / "tools" / f"{tool}.py").exists()
    assert tools[tool]["file"] == "deep_reasoning.app_tools"
    assert tools[tool]["description"] == description
    assert "library.tool_shadowed" in {entry["event"] for entry in logs}


def test_the_engine_config_builds_a_registry_and_a_reasoner_without_a_model_call(
    library, advisors, waitlist, tmp_path, monkeypatch, no_network
):
    import_into(library, waitlist)
    import_into(library, advisors)
    config = run_config(library, tmp_path, "course_advisor")
    registry = build_registry(
        inline=list(config.namespaces.values()), default_repl=config.repl
    )
    lookup = registry.resolve("course_advisor").decompositions
    assert [each.name for each in lookup] == [
        *(each.name for each in HAND_OFFS),
        "catalog lookup",
    ]
    assert registry.resolve("course_advisor").tools == ["llm"]
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.chdir(tmp_path)
    reasoner, _alias = build_reasoner(config, run_dir=tmp_path / "run")
    assert isinstance(reasoner, DeepReasoner)
    assert config.entry_namespace == "course_advisor"
    close_run(reasoner)


def change(library, act, *args):
    with closing(open_library(library)) as connection:
        return act(connection, *args)


def registry_of(library, tmp_path):
    return build_namespace_registry(run_config(library, tmp_path))


def decomposition_rows(library):
    with closing(open_library(library)) as connection:
        rows = connection.execute("SELECT namespace FROM decompositions")
        return [namespace for (namespace,) in rows]


def test_a_namespace_is_added_at_the_top_and_its_child_inherits_from_it(
    library, tmp_path
):
    change(library, add_namespace, "research", RESEARCH)
    change(library, add_namespace, "research.papers", "vars: {year: 2026}\n")
    registry = registry_of(library, tmp_path)
    papers = registry.resolve("research.papers")
    assert registry.chain("research.papers") == [ROOT, "research", "research.papers"]
    assert papers.vars == {"topic": "ml", "year": 2026}
    assert papers.sources.vars == {"topic": "research", "year": "research.papers"}
    assert papers.system_suffix == "Cite each source.\n"


@pytest.mark.parametrize("text", [f"# why\n{RESEARCH}", "# fill in later\n"])
def test_a_namespace_keeps_its_yaml_as_written(library, text):
    change(library, add_namespace, "research", text)
    with closing(open_library(library)) as connection:
        [shown] = [
            each
            for each in panel(connection)["namespaces"]
            if each["name"] == "research"
        ]
    assert shown["yaml"] == text
    assert shown["decompositions"] == []


def test_a_namespace_whose_parent_is_missing_is_refused(library):
    before = dumped(library)
    with pytest.raises(LibraryError, match='Add namespace "research" first'):
        change(library, add_namespace, "research.papers", RESEARCH)
    assert dumped(library) == before


@pytest.mark.parametrize(
    ("name", "text", "problem"),
    [
        ("", RESEARCH, "Name the namespace"),
        ("research papers", RESEARCH, "A namespace name is words of letters"),
        ("research.", RESEARCH, "A namespace name is words of letters"),
        ("router", RESEARCH, 'There is already a namespace "router"'),
        ("research", "tools: [llm\n", "The YAML doesn't parse (line 2:"),
        ("research", "- llm\n", "The YAML should be a mapping of fields"),
        (
            "research",
            "name: research\n",
            'Remove "name": a namespace is named in the name field',
        ),
        ("research", "decompositions: []\n", "added in the Decompositions tab"),
        (
            "research",
            "tool: [llm]\n",
            'Remove "tool": a namespace\'s YAML holds only repl,',
        ),
        ("research", "tools: llm\n", "tools should be a list of tool names"),
        ("research", "system_suffix: 5\n", 'write "system_suffix: |"'),
    ],
)
def test_a_namespace_the_panel_cannot_take_is_refused_with_a_plain_message(
    imported, name, text, problem
):
    before = dumped(imported)
    with pytest.raises(LibraryError, match=re.escape(problem)):
        change(imported, add_namespace, name, text)
    assert dumped(imported) == before


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "tools: [ghost]\n",
            "The engine refuses it: namespace 'research' grants unknown tool 'ghost'",
        ),
        (
            "spawn: [nowhere]\n",
            (
                'spawn names "nowhere", which is not a namespace in the Library: add'
                " nowhere first, or take it out of spawn."
            ),
        ),
    ],
)
def test_a_namespace_granting_an_unknown_tool_or_spawning_nowhere_is_not_written(
    imported, text, problem
):
    before = dumped(imported)
    with pytest.raises(LibraryError, match=re.escape(problem)):
        change(imported, add_namespace, "research", text)
    with pytest.raises(
        LibraryError, match=re.escape(problem.replace("research", "router"))
    ):
        change(imported, change_namespace, "router", text)
    assert dumped(imported) == before


def test_a_new_namespace_spawning_one_to_go_under_it_says_in_which_order_to_add_them(
    imported,
):
    before = dumped(imported)
    refusal = (
        'spawn names "team.a", which goes under team and isn\'t in the Library yet: add'
        " team without it first, then team.a, then put team.a in the spawn of team."
    )
    with pytest.raises(LibraryError, match=re.escape(refusal)):
        change(imported, add_namespace, "team", "spawn: [team.a]\n")
    assert dumped(imported) == before


def test_an_edit_replaces_the_fields_and_keeps_the_decompositions(imported, tmp_path):
    change(imported, change_namespace, "router", "system_suffix: Be brief.\n")
    router = run_config(imported, tmp_path).namespaces["router"]
    assert router.spawn is None
    assert router.system_suffix == "Be brief."
    assert [each.name for each in router.decompositions] == ROUTER_DECOMPOSITIONS


def test_a_namespace_may_hold_a_date_in_its_vars(imported, tmp_path):
    change(imported, add_namespace, "research", "vars: {deadline: 2026-11-01}\n")
    research = run_config(imported, tmp_path).namespaces["research"]
    assert research.vars["deadline"] == date(2026, 11, 1)


def test_root_can_be_edited(imported, tmp_path):
    change(imported, change_namespace, ROOT, "tools: [llm]\nvars: {term: fall}\n")
    registry = registry_of(imported, tmp_path)
    assert registry.resolve("course_advisor").vars["term"] == "fall"


@pytest.mark.parametrize("text", ["", "{}\n", "# nothing yet\n"])
def test_a_root_saved_blank_stays_blank_when_the_library_opens_again(
    library, tmp_path, text
):
    change(library, change_namespace, ROOT, text)
    assert run_config(library, tmp_path).namespaces[ROOT].tools == []


def test_deleting_a_namespace_removes_it_and_its_decompositions(imported, tmp_path):
    change(imported, add_namespace, "research", RESEARCH)
    change(imported, add_decomposition, "research", ADDED)
    change(imported, delete_namespace, "research", 1)
    assert set(run_config(imported, tmp_path).namespaces) == IMPORTED_NAMESPACES
    assert "research" not in decomposition_rows(imported)
    assert decomposition_rows(imported).count("router") == len(ROUTER_DECOMPOSITIONS)


def test_deleting_the_default_namespace_starts_new_chats_in_root(imported):
    change(imported, delete_namespace, "router", len(ROUTER_DECOMPOSITIONS))
    assert default_namespace(imported) == ROOT
    assert "router" not in decomposition_rows(imported)


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("research", "Delete its namespaces first: research.papers."),
        ("course_advisor", "First take course_advisor out of the spawn of router"),
        (ROOT, "root can't be deleted"),
        ("nowhere", 'The Library has no namespace "nowhere" to delete.'),
    ],
)
def test_a_refused_delete_says_why_and_changes_nothing(imported, name, problem):
    change(imported, add_namespace, "research", RESEARCH)
    change(imported, add_namespace, "research.papers", "")
    before = dumped(imported)
    with pytest.raises(LibraryError, match=re.escape(problem)):
        change(imported, delete_namespace, name, 0)
    assert dumped(imported) == before


def test_each_namespace_row_gives_the_refusal_its_delete_would_get(imported):
    change(imported, add_namespace, "research", RESEARCH)
    change(imported, add_namespace, "research.papers", "")
    for row in panel_of(imported)["namespaces"]:
        if row["refusal"] is None:
            continue
        with pytest.raises(LibraryError) as refused:
            change(imported, delete_namespace, row["name"], len(row["decompositions"]))
        assert str(refused.value) == row["refusal"]
    refusals = {row["name"]: row["refusal"] for row in panel_of(imported)["namespaces"]}
    assert (refusals["research.papers"], refusals["router"]) == (None, None)
    assert {"research", "course_advisor", ROOT} <= {
        name for name, refusal in refusals.items() if refusal
    }


def test_a_delete_that_counted_fewer_decompositions_than_are_there_is_refused(
    imported,
):
    change(imported, add_namespace, "research", RESEARCH)
    change(imported, add_decomposition, "research", ADDED)
    before = dumped(imported)
    with pytest.raises(LibraryError, match="research has 1 decomposition now, not 0"):
        change(imported, delete_namespace, "research", 0)
    assert dumped(imported) == before


def test_a_decomposition_added_while_its_namespace_is_deleted_is_refused(imported):
    change(imported, add_namespace, "research", RESEARCH)
    with ThreadPoolExecutor(1) as adding:
        with closing(open_library(imported)) as connection, writing(connection):
            connection.execute("DELETE FROM namespaces WHERE name = 'research'")
            added = adding.submit(
                change, imported, add_decomposition, "research", ADDED
            )
            # Long enough for it to find research there, then wait for the write lock.
            time.sleep(WAIT_FOR_THE_LOCK_S)
        with pytest.raises(LibraryError, match='no namespace "research"'):
            added.result()
    assert "research" not in decomposition_rows(imported)


@pytest.mark.parametrize(
    ("messages", "problems"),
    [(case["messages"], case["problems"]) for case in TURN_CASES],
    ids=[case["case"] for case in TURN_CASES],
)
def test_a_decompositions_turns_are_checked_as_a_worked_example(messages, problems):
    assert turn_problems(messages) == problems


def test_the_seeded_examples_and_the_engines_own_namespace_examples_pass(library):
    for config in ("catalog/advisors.yaml", "examples/experience/experience.yaml"):
        import_into(library, CONFIGS / config)
    with closing(open_library(library)) as connection:
        namespaces = panel(connection)["namespaces"]
    rows = [
        (each["name"], each["problems"])
        for row in namespaces
        for each in row["decompositions"]
    ]
    # the sixteen seeded, advisors.yaml's five, and experience.yaml's two, one with a message after its answer
    assert len(rows) == 23
    assert rows == [(name, []) for name, _ in rows]


def test_a_decomposition_with_problems_is_stored_and_the_panel_lists_them(library):
    stored, _ = change(library, add_decomposition, ROOT, NO_THINK)
    with closing(open_library(library)) as connection:
        [root, *_] = panel(connection)["namespaces"]
    no_think = ["Turn 2 (assistant) has no <think>…</think>."]
    assert stored == {
        "namespace": ROOT,
        "name": "no think",
        "yaml": NO_THINK,
        "problems": no_think,
    }
    assert [(each["name"], each["problems"]) for each in root["decompositions"]] == [
        *SEEDED_ROWS,
        ("no think", no_think),
    ]


def test_a_shared_files_decomposition_the_panel_would_refuse_shows_why_instead(
    library,
):
    blank = "name: ' '\nmessages: [{role: user, content: hi}]\n"
    with closing(open_library(library)) as connection:
        with writing(connection):
            connection.execute(UPSERT_DECOMPOSITION, (ROOT, " ", blank))
            # Under a namespace no longer there, so the engine never reads it.
            connection.execute(UPSERT_DECOMPOSITION, ("gone", "x", "just text\n"))
        [root, *_] = panel(connection)["namespaces"]
    assert root["decompositions"][-1]["problems"] == [
        'The name should be text, like "name: count the open tickets".'
    ]


def test_a_shared_files_decompositions_under_a_root_it_lacks_are_still_checked(library):
    with closing(open_library(library)) as connection:
        with writing(connection):
            connection.execute("DELETE FROM namespaces WHERE name = ?", (ROOT,))
            connection.execute(UPSERT_DECOMPOSITION, (ROOT, "no think", NO_THINK))
        [root, *_] = panel(connection)["namespaces"]
    assert [(each["name"], each["problems"]) for each in root["decompositions"]] == [
        *SEEDED_ROWS,
        ("no think", ["Turn 2 (assistant) has no <think>…</think>."]),
    ]


def names_in(library, namespace):
    with closing(open_library(library)) as connection:
        rows = connection.execute(
            "SELECT name FROM decompositions WHERE namespace = ? ORDER BY rowid",
            (namespace,),
        )
        return [name for (name,) in rows]


def panel_of(library):
    with closing(open_library(library)) as connection:
        return panel(connection)


def test_a_decomposition_is_edited_in_its_place_and_may_take_a_new_name(imported):
    [first, *rest] = ROUTER_DECOMPOSITIONS
    renamed = NO_THINK.replace("no think", "route it")
    stored = dict(names_and_yaml(imported, "router"))[first]
    change(imported, edit_decomposition, "router", first, stored, renamed)
    assert names_in(imported, "router") == ["route it", *rest]
    [router] = [
        each for each in panel_of(imported)["namespaces"] if each["name"] == "router"
    ]
    assert router["decompositions"][0] == {
        "name": "route it",
        "yaml": renamed,
        "problems": ["Turn 2 (assistant) has no <think>…</think>."],
    }


def names_and_yaml(library, namespace):
    with closing(open_library(library)) as connection:
        return connection.execute(
            "SELECT name, yaml FROM decompositions WHERE namespace = ? ORDER BY rowid",
            (namespace,),
        ).fetchall()


def test_roots_decompositions_are_edited_and_deleted_by_the_user(library):
    [(name, text), *rest] = names_and_yaml(library, ROOT)
    edited = text.replace("content:", "content:  ", 1)
    change(library, edit_decomposition, ROOT, name, text, edited)
    change(library, delete_decomposition, ROOT, name, edited)
    assert names_and_yaml(library, ROOT) == rest


def test_an_edit_may_not_take_the_name_of_another_decomposition_there(imported):
    [first, second, _] = names_and_yaml(imported, "router")
    before = dumped(imported)
    with pytest.raises(LibraryError, match=f'router already has "{second[0]}"'):
        change(
            imported,
            edit_decomposition,
            "router",
            first[0],
            first[1],
            second[1],
        )
    assert dumped(imported) == before


BY_WHOM = "(by the librarian, another tab or an import)"
NOT_SAVED = (
    f"was changed after you opened it {BY_WHOM}, so nothing was saved: saving would undo"
    " that change. Look at it as it is now, then save again."
)


@pytest.mark.parametrize(
    ("act", "changed", "gone"),
    [
        (
            lambda connection, name, was: edit_decomposition(
                connection, "router", name, was, ADDED
            ),
            NOT_SAVED,
            NOT_SAVED,
        ),
        (
            lambda connection, name, was: delete_decomposition(
                connection, "router", name, was
            ),
            (
                f"was changed after you opened it {BY_WHOM}, so it wasn't deleted. Look"
                " at it as it is now, then delete it again if you still want to."
            ),
            f"was already deleted or renamed {BY_WHOM}.",
        ),
    ],
    ids=["edit", "delete"],
)
def test_an_edit_or_delete_of_a_decomposition_changed_or_deleted_meanwhile_is_refused(
    imported, act, changed, gone
):
    [(name, text), *_] = names_and_yaml(imported, "router")
    change(imported, add_decomposition, "router", text + "# changed\n", True)
    before = dumped(imported)
    with pytest.raises(
        ChangedMeanwhile, match=re.escape(f'"{name}" in router {changed}')
    ):
        change(imported, act, name, text)
    with pytest.raises(ChangedMeanwhile, match=re.escape(f'"gone" in router {gone}')):
        change(imported, act, "gone", text)
    assert dumped(imported) == before


def test_a_namespace_saved_over_a_change_made_meanwhile_is_refused(imported):
    change(imported, add_namespace, "research", RESEARCH)
    change(imported, change_namespace, "research", "vars: {topic: nlp}\n")
    before = dumped(imported)
    with pytest.raises(ChangedMeanwhile, match="Namespace research was changed"):
        change(imported, change_namespace, "research", "", None, RESEARCH)
    assert dumped(imported) == before
    change(imported, change_namespace, "research", "", None, "vars: {topic: nlp}\n")
    # One that sets nothing opens blank, as the panel shows it.
    change(imported, change_namespace, "research", "tools: [llm]\n", None, "")


TOOL_SOURCE = "def build(client, params):\n    return None\n"


def test_a_tool_is_deleted_only_once_no_namespace_grants_it(library):
    with closing(open_library(library)) as connection:
        block = {"factory": "build", "factory_from": "tally.py"}
        store_tool(connection, "tally", block, TOOL_SOURCE)
        add_namespace(connection, "counting", "tools: [tally]\n")
        add_namespace(connection, "counting.more", "tools: [tally]\n")
    before = dumped(library)
    with pytest.raises(
        LibraryError, match=re.escape("tally is granted in counting, counting.more:")
    ):
        change(library, delete_tool, "tally")
    assert dumped(library) == before
    [tally] = [each for each in panel_of(library)["tools"] if each["name"] == "tally"]
    assert (tally["stored"], tally["source"]) == (True, TOOL_SOURCE)
    assert tally["refusal"] == (
        "tally is granted in counting, counting.more: take it out of their tools first."
    )
    change(library, delete_namespace, "counting.more", 0)
    change(library, change_namespace, "counting", "")
    [tally] = [each for each in panel_of(library)["tools"] if each["name"] == "tally"]
    assert tally["refusal"] is None
    change(library, delete_tool, "tally")
    assert "tally" not in [each["name"] for each in panel_of(library)["tools"]]


@pytest.mark.parametrize(
    ("name", "refusal"),
    [
        ("llm", "llm is granted in root: take it out of their tools first."),
        ("history", "history is granted in historian: take it out of their tools"),
        ("rag", 'The Library holds no tool "rag" to delete.'),
    ],
)
def test_a_tool_the_library_holds_no_row_for_is_not_deleted(library, name, refusal):
    shown = {each["name"]: each for each in panel_of(library)["tools"]}
    assert all(not each["stored"] and each["source"] is None for each in shown.values())
    before = dumped(library)
    with pytest.raises(LibraryError, match=re.escape(refusal)) as refused:
        change(library, delete_tool, name)
    assert dumped(library) == before
    # rag, granted nowhere, is not listed at all.
    assert name not in shown or shown[name]["refusal"] == str(refused.value)


def test_a_run_setting_the_engine_doesnt_read_is_deleted_and_one_it_reads_stays(
    library,
):
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(UPSERT_SETTING, ("catalog_path", dump("credits.yaml")))
    rows = {each["label"]: each for each in panel_of(library)["run_settings"]}
    assert rows["catalog_path"] == {
        "label": "catalog_path",
        "items": [{"value": "credits.yaml", "source": UNREAD_SETTING}],
        "removable": True,
    }
    assert not rows["max_iter"]["removable"]
    before = dumped(library)
    with pytest.raises(LibraryError, match="system_prompt is a run setting the engine"):
        change(library, delete_setting, "system_prompt")
    assert dumped(library) == before
    change(library, delete_setting, "catalog_path")
    labels = [each["label"] for each in panel_of(library)["run_settings"]]
    assert "catalog_path" not in labels
    with pytest.raises(LibraryError, match='no run setting "catalog_path"'):
        change(library, delete_setting, "catalog_path")


# A worked example whose answer cell is given as *code*, in namespace-free YAML.
def answering(name, code):
    return (
        f"name: {name}\nmessages:\n- role: user\n  content: What is the total?\n"
        "- role: assistant\n  content: |\n    <think>Print it, then answer.</think>\n"
        f"    <repl>\n    total = 3\n    {code}\n    FinalAnswer(total)\n    </repl>\n"
    )


TURN_2_TEMPLATE = (
    "Turn 2 (assistant), line 4: the engine reads text holding {{ or {% as a Jinja"
    " template"
)
# Templates no run renders: each with why, as the check words it.
INCLUDING = "includes another template, which a run can't"


def reaching(name):
    return f"reaches into Python internals ({name}), which a run refuses"


REFUSED_TEMPLATES = {
    "an-attribute-of-the-task": (
        'print("{{ task.__class__ }}")',
        reaching("__class__"),
    ),
    "an-attribute-of-a-jinja-global": (
        'print("{{ cycler.__init__ }}")',
        reaching("__init__"),
    ),
    "an-unsafe-name": ('print("{{ funcs.llm.mro }}")', reaching("mro")),
    "a-key": ("print(\"{{ vars['_hidden'] }}\")", reaching("_hidden")),
    "an-include": ("{% include 'x.html' %}", INCLUDING),
    "an-extends": ("{% extends 'x.html' %}", INCLUDING),
    "an-import": ("{% import 'x.html' as x %}", INCLUDING),
    "a-from-import": ("{% from 'x.html' import x %}", INCLUDING),
}
# Templates that would stop the panel if rendered or compiled unguarded.
NESTED_TOO_DEEP = "{% for a in [1] %}" * 21 + "x" + "{% endfor %}" * 21
BILLIONS_OF_LOOPS = (
    "{% for i in range(99999) %}{% for j in range(99999) %}{% endfor %}{% endfor %}"
)


@pytest.mark.parametrize(
    ("code", "problem"),
    [
        (
            'print(f"{{{total}}}")',
            (
                "Turn 2 (assistant), line 4: the engine reads text holding {{ or {% as a"
                " Jinja template, and this one doesn't parse (expected token ':', got '}')."
            ),
        ),
        (
            'print("{{ total }}")',
            (
                "Turn 2 (assistant), line 4: the engine reads text holding {{ or {% as a"
                ' Jinja template, and this one names "total"; a template may name only'
                " task, vars, funcs, coroutines and the run settings'"
                " prompt_template_variables, which the engine fills in."
            ),
        ),
        (
            'print("{{ task }}")  {# a note',
            (
                "Turn 2 (assistant), line 4: the engine reads text holding {{ or {% as a"
                " Jinja template, and this one doesn't parse (Missing end of comment tag)."
            ),
        ),
        (
            NESTED_TOO_DEEP,
            (
                "Turn 2 (assistant): the engine reads text holding {{ or {% as a Jinja"
                " template, and this one can't be compiled (too many statically nested"
                " blocks)."
            ),
        ),
        (
            BILLIONS_OF_LOOPS,
            'Turn 2 (assistant), line 4: the engine reads text holding {{ or {% as a Jinja template, and this one names "range"',
        ),
        (
            "{{ lipsum(n=100000) }}",
            'Turn 2 (assistant), line 4: the engine reads text holding {{ or {% as a Jinja template, and this one names "lipsum"',
        ),
        *(
            (code, f"{TURN_2_TEMPLATE}, and this one {why}.")
            for code, why in REFUSED_TEMPLATES.values()
        ),
    ],
    ids=[
        "an-f-string",
        "an-unknown-name",
        "a-comment",
        "nested-too-deep",
        "billions-of-loops",
        "a-jinja-global",
        *REFUSED_TEMPLATES,
    ],
)
def test_a_decomposition_the_engine_cant_render_is_refused_naming_its_turn(
    library, code, problem
):
    text = answering("braces", code)
    before = dumped(library)
    escape = "Write {{ '{{' }} for a literal {{ and {{ '{%' }} for {%"
    with pytest.raises(LibraryError, match=re.escape(problem)) as refused:
        change(library, add_decomposition, ROOT, text)
    assert escape in str(refused.value)
    [(name, stored), *_] = names_and_yaml(library, ROOT)
    with pytest.raises(LibraryError, match=re.escape(problem)):
        change(library, edit_decomposition, ROOT, name, stored, text)
    assert dumped(library) == before


@pytest.mark.parametrize(
    ("namespace", "code", "rendered"),
    [
        (ROOT, 'print("{{ task }}")', 'print("Count the tickets.")'),
        (ROOT, "print(f\"{{ '{{' }}{total}}}\")", 'print(f"{{{total}}}")'),
        (ROOT, '{% raw %}print(f"{{{total}}}"){% endraw %}', 'print(f"{{{total}}}")'),
        (ROOT, 'print("{{ term }}")', 'print("fall")'),
        (
            "research",
            'print("{{ vars.topic.value }}", "{{ funcs.llm.name }}")',
            'print("ml", "llm")',
        ),
    ],
    ids=["the-task", "escaped", "raw", "a-template-variable", "a-namespaces-var"],
)
def test_a_template_the_check_passes_is_stored_and_the_engine_renders_it(
    library, tmp_path, monkeypatch, no_network, namespace, code, rendered
):
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(
            UPSERT_SETTING, ("prompt_template_variables", dump({"term": "fall"}))
        )
    change(library, add_namespace, "research", RESEARCH)
    change(library, add_decomposition, namespace, answering("braces", code))
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.chdir(tmp_path)
    config = run_config(library, tmp_path, namespace)
    reasoner, _alias = build_reasoner(config, run_dir=tmp_path / "run")
    # The engine renders an agent's worked examples as it starts, before any model call.
    reasoner.send("Count the tickets.")._start()
    examples = reasoner.ctx.examples
    close_run(reasoner)
    [_, answer] = [each["content"] for each in examples if each["example"] == "braces"]
    assert rendered in answer


def test_a_shared_files_decomposition_the_engine_cant_render_is_marked_with_why(
    library, tmp_path, monkeypatch, no_network
):
    stored = {
        "braces": 'print(f"{{{total}}}")',
        "nested": NESTED_TOO_DEEP,
        "loops": BILLIONS_OF_LOOPS,
    } | {name: code for name, (code, _) in REFUSED_TEMPLATES.items()}
    with closing(open_library(library)) as connection, writing(connection):
        for name, code in stored.items():
            connection.execute(
                UPSERT_DECOMPOSITION, (ROOT, name, answering(name, code))
            )
    started = time.monotonic()
    [root, *_] = panel_of(library)["namespaces"]
    # Read, never rendered: billions of loops cost nothing.
    assert time.monotonic() - started < PANEL_READ_S
    marked = {each["name"]: each["problems"] for each in root["decompositions"]}
    for name in stored:
        [problem] = marked[name]
        assert problem.startswith("Turn 2 (assistant)")
        assert "the engine reads text holding {{ or {% as a Jinja template" in problem
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.chdir(tmp_path)
    reasoner, _alias = build_reasoner(run_config(library, tmp_path), run_dir=tmp_path)
    with pytest.raises(ValueError, match="'braces' in namespace 'root' could not be"):
        reasoner.send("Count the tickets.")._start()
    close_run(reasoner)


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("count it ", 'The name "count it " starts or ends with a space'),
        (" count it", 'The name " count it" starts or ends with a space'),
        ("count\u200bit", "The name holds a character you can't see (U+200B)"),
    ],
)
def test_a_decomposition_named_unlike_it_looks_is_refused(library, name, problem):
    text = ADDED.replace("name: added", f"name: {json.dumps(name)}")
    with pytest.raises(LibraryError, match=re.escape(problem)):
        change(library, add_decomposition, ROOT, text)


def test_a_decomposition_past_a_million_characters_is_refused(library):
    text = ADDED + "#" * 1_000_000 + "\n"
    with pytest.raises(
        LibraryError,
        match=f"may be at most 1,000,000 characters; this has {len(text):,}",
    ):
        change(library, add_decomposition, ROOT, text)
    [(name, stored), *_] = names_and_yaml(library, ROOT)
    with pytest.raises(LibraryError, match="may be at most 1,000,000 characters"):
        change(library, edit_decomposition, ROOT, name, stored, text)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "reasoner: {type: chatgpt}\n",
            (
                "The engine refuses it: namespace 'research': unknown reasoner type"
                " 'chatgpt'; known: 'chat', 'claude_code'."
            ),
        ),
        ("vars: {max: 10}\n", 'Var "max" would hide Python\'s own max'),
        ("vars: {type: memo}\n", 'Var "type" would hide Python\'s own type'),
        ("vars: {1abc: 1}\n", 'Var "1abc" needs a Python name, like "year"'),
        ("vars: {my var: 1}\n", 'Var "my var" needs a Python name'),
        ("vars: {class: 1}\n", 'Var "class" is a Python keyword'),
        (
            "vars: {_hidden: 1}\n",
            (
                'Var "_hidden" starts with _, which templates may not read; rename it,'
                ' like "hidden".'
            ),
        ),
        (
            "system_suffix: 'Answer for {{ dept }}.'\n",
            (
                "The system_suffix of research, line 1: the engine reads text holding {{ or {% as a"
                ' Jinja template, and this one names "dept"'
            ),
        ),
        (
            "system_suffix: 'Answer as {{ task.__class__ }}.'\n",
            (
                "The system_suffix of research, line 1: the engine reads text holding {{ or {% as a"
                " Jinja template, and this one reaches into Python internals"
                " (__class__), which a run refuses."
            ),
        ),
        ("vars: {1: a}\n", 'A var\'s name should be text, like "year", not 1.'),
        (
            "vars: {llm: 1}\n",
            (
                'In research, var "llm" (set in research) has the name of the tool llm'
                " (granted in root), which takes its place: rename the var."
            ),
        ),
    ],
)
def test_a_namespace_that_would_fail_its_runs_or_hide_a_var_is_refused(
    library, text, problem
):
    change(library, add_namespace, "drafts", "")
    before = dumped(library)
    with pytest.raises(LibraryError, match=re.escape(problem)):
        change(library, add_namespace, "research", text)
    with pytest.raises(
        LibraryError, match=re.escape(problem.replace("research", "drafts"))
    ):
        change(library, change_namespace, "drafts", text)
    assert dumped(library) == before


def test_a_tool_granted_below_a_var_of_its_name_is_refused(library):
    tool = {"factory": "build", "factory_from": "tally.py"}
    with closing(open_library(library)) as connection:
        store_tool(connection, "tally", tool, "def build(client, params):\n    pass\n")
    change(library, add_namespace, "research", "vars: {tally: 3}\n")
    change(library, add_namespace, "research.papers", "")
    before = dumped(library)
    with pytest.raises(
        LibraryError,
        match=re.escape('In research.papers, var "tally" (set in research)'),
    ):
        change(library, change_namespace, "research.papers", "tools: [tally]\n")
    assert dumped(library) == before


@pytest.mark.parametrize("name", ["research ", "research\u200b", " research"])
def test_a_namespace_named_unlike_it_looks_is_refused(library, name):
    with pytest.raises(LibraryError, match="A namespace name is words of letters"):
        change(library, add_namespace, name, "")


def test_a_namespace_granting_a_tool_root_grants_says_it_grants_it_here(library):
    change(library, add_namespace, "research", "tools: [llm]\nvars: {draft: null}\n")
    [research] = [
        each for each in panel_of(library)["namespaces"] if each["name"] == "research"
    ]
    fields = {each["label"]: each["items"] for each in research["fields"]}
    assert fields["Tools"][0] == {
        "value": "llm",
        "source": "granted here",
        "here": True,
    }
    assert fields["Variables"] == [
        {"value": "draft (None)", "source": "set here", "here": True}
    ]


def test_the_names_a_template_may_read_are_those_the_engine_fills_in():
    assert set(FILLED_IN) == set(_render_context([], {}, "a task"))


# Templates as the engine's docs and the panel's template write them, each rendered
# against TICKETS, a topic var and a word_count tool.
WRITTEN_TEMPLATES = {
    "the task": ("Count the tickets: {{ task }}", "Count the tickets: open tickets"),
    "a filter": ("{{ task|upper }}", "OPEN TICKETS"),
    "a var": ("{{ vars.topic.value }}", "ml"),
    "each tool": (
        "{% for f in funcs.values() %}{{ f.name }} {% endfor %}",
        "word_count ",
    ),
    "a loop": (
        "{% for word in task.split() %}{{ loop.index }}:{{ word }} {% endfor %}",
        "1:open 2:tickets ",
    ),
    "a template variable": ("{{ term }}", "fall"),
    "a list it builds": (
        "{% set seen = [] %}{{ seen.append(task) }} {{ seen|join }}",
        "None open tickets",
    ),
}
TICKETS = "open tickets"


def test_each_worked_example_a_run_renders_renders_the_same_sandboxed(
    library, tmp_path, monkeypatch
):
    for config in ("catalog/advisors.yaml", "examples/experience/experience.yaml"):
        import_into(library, CONFIGS / config)
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(
            UPSERT_SETTING, ("prompt_template_variables", dump({"term": "fall"}))
        )
    for name, (template, _) in WRITTEN_TEMPLATES.items():
        example = {"name": name, "messages": [{"role": "user", "content": template}]}
        change(library, add_decomposition, ROOT, dump(example))
    registry = registry_of(library, tmp_path)
    bindings = [var("topic", "ml", "a topic"), func("word_count", len, "words")]
    context = _render_context(bindings, {"term": "fall"}, TICKETS)

    def rendered():
        return {
            (namespace, example.name): _rendered_turns(example, context, namespace)
            for namespace in registry.names
            for example in registry.resolve(namespace).decompositions
        }

    unsandboxed = rendered()
    # restored after the test
    monkeypatch.setattr(deep_reasoner.llm, "_JinjaEnv", deep_reasoner.llm._JinjaEnv)
    sandbox_templates()
    assert rendered() == unsandboxed
    for name, (_, expected) in WRITTEN_TEMPLATES.items():
        assert unsandboxed[ROOT, name] == [("user", expected)]


def test_a_reasoner_the_engine_doesnt_know_elsewhere_holds_up_no_other_change(library):
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(UPSERT_NAMESPACE, ("odd", "reasoner: {type: chatgpt}\n"))
    change(library, add_namespace, "mine", "")
    change(library, change_namespace, "mine", "system_suffix: Be brief.\n")
    change(library, delete_namespace, "mine", 0)
    with pytest.raises(LibraryError, match="namespace 'odd': unknown reasoner type"):
        change(library, add_namespace, "odd.child", "")
    with pytest.raises(LibraryError, match="namespace 'odd': unknown reasoner type"):
        change(library, change_namespace, "odd", "reasoner: {type: chatgpt}\n")
