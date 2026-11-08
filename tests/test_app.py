import itertools
import json
import re
import shutil
import socket
import sqlite3
import stat
import threading
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pytest
import yaml
from deep_reasoner.v2.agent import DEFAULT_EXAMPLE_FENCE
from fake_model import (
    ASKED,
    CHANGED_SUFFIX,
    EXAMPLE,
    RESEARCH,
    RESEARCH_CHANGED,
    REWORDED,
    REWORDED_ASK,
    SERVER_KEY,
    SERVER_NAME,
    SPLIT_COUNT,
    SUFFIX,
    WORD_COUNT,
    WORD_COUNT_DOC,
    FakeOpenAI,
    folded_answer,
    library_cell,
    repl,
    scripted,
    task_and_turn,
    write,
)
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from deep_reasoning import settings as settings_module
from deep_reasoning.app import NO_PAGE, WEB, create_app
from deep_reasoning.chats import TITLE_CHARS, create_chat
from deep_reasoning.folders import data_folder
from deep_reasoning.importing import import_config
from deep_reasoning.library import (
    GRANT_IGNORED,
    HAND_OFF_TURNS,
    HISTORIAN,
    HISTORIAN_SUFFIX,
    HISTORIANS_EXAMPLES,
    HISTORY_DESCRIPTION,
    LIBRARIANS_EXAMPLES,
    MAX_YAML_CHARS,
    PAGE_DESCRIPTION,
    ROOT,
    TOO_LONG_ROW,
    UNREAD_DECOMPOSITION,
    UPSERT_DECOMPOSITION,
    UPSERT_NAMESPACE,
    UPSERT_SETTING,
    UPSERT_TOOL,
    YAML_NOT_TEXT,
    dump,
    hand_off,
    open_library,
    parsed,
    running_code,
    writing,
)
from deep_reasoning.pages import add_page
from deep_reasoning.runs import CHANGE_FIELDS, EARLIER_TURNS_NOTE, LOST, save_event
from deep_reasoning.settings import (
    DEFAULT_FOLDER,
    ENV_FILE,
    LAST_MODEL,
    SERVERS,
    VENDORS,
    add_server,
    default_folder,
    saved_keys,
    server_key_env,
)

pytestmark = pytest.mark.usefixtures("no_vendor_keys")

LOCAL = "http://127.0.0.1"
SAVED_KEY = "sk-saved-in-settings"
SHELL_KEY = "sk-from-the-shell"
VLLM = "http://127.0.0.1:8001/v1"
VALID = """name: next course
messages:
- role: user
  content: What comes after CS201?
- role: assistant
  content: <repl>FinalAnswer("CS310")</repl>
"""


HELD_S = 30
PLAN = {
    "Count to two.": [repl("print(1)"), repl("FinalAnswer(2)")],
    "What did I ask?": [repl("FinalAnswer('to count')")],
}


@pytest.fixture
def client(imported, home):
    return TestClient(create_app(imported, home), base_url=LOCAL)


@pytest.fixture
def data(home, library):
    """The Library file's data folder in the app's home: its runs and scratch folders."""
    return data_folder(home, library)


@pytest.fixture
def connect(home):
    """An HTTP client of the app over a Library: call it with the Library."""

    def to(library):
        transport = httpx.ASGITransport(app=create_app(library, home))
        return httpx.AsyncClient(transport=transport, base_url=LOCAL)

    return to


def server_sent(text):
    """(id, data) of each event in an event stream; keep-alive comments skipped."""
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        if "data" in fields:
            events.append((int(fields["id"]), json.loads(fields["data"])))
    return events


async def new_chat(api):
    """A new chat's id."""
    response = await api.post("/api/chats")
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def send(api, chat, prompt):
    """Start a run of *prompt* in *chat*; its id."""
    response = await api.post(f"/api/chats/{chat}/runs", json={"prompt": prompt})
    assert response.status_code == 201, response.text
    return response.json()["run"]


async def run_events(api, run):
    """Every event of the run, once it has ended."""
    response = await api.get(f"/api/runs/{run}/events")
    return [event for _, event in server_sent(response.text)]


def row(seq, kind, **fields):
    """An event's (seq, data) row, as the app writes it."""
    return seq, json.dumps({"seq": seq, "t": 100.0 + seq, "kind": kind} | fields)


def plant(library, chat, run, rows):
    """A run of *chat* in the app file, holding the (seq, data) rows given as they are,
    as a file from someone else might."""
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute("INSERT INTO runs VALUES (?, ?)", (run, chat))
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?)", [(run, *each) for each in rows]
        )


def ends_saved(library, run):
    with closing(open_library(library)) as connection:
        found = connection.execute(
            "SELECT count(*) FROM events WHERE run = ? AND data LIKE '%turn.end%'",
            (run,),
        )
        return found.fetchone()[0]


def saved_runs(library):
    """The runs whose events the app file holds."""
    with closing(open_library(library)) as connection:
        return {run for (run,) in connection.execute("SELECT DISTINCT run FROM events")}


def namespaces(client):
    return {
        each["name"]: each for each in client.get("/api/library").json()["namespaces"]
    }


def test_the_namespace_menu_reads_the_panels_namespaces_and_default(client):
    panel = client.get("/api/library").json()
    menu = client.get("/api/namespaces").json()
    assert menu == {
        "namespaces": [
            {"name": each["name"], "usable": True}
            for each in panel["namespaces"]
            if each["usable"]
        ],
        "default_namespace": panel["default_namespace"],
    }


def field(namespace, label):
    return next(each["items"] for each in namespace["fields"] if each["label"] == label)


def test_root_serves_the_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Deep Reasoning</title>" in response.text


@pytest.mark.parametrize(
    "path",
    [
        "app.js",
        "fold.js",
        "tree.js",
        "style.css",
        "library.js",
        "settings.js",
        "page.js",
        "page.html",
    ],
)
def test_static_assets_are_served(client, path):
    response = client.get(f"/{path}")
    assert response.status_code == 200
    assert response.content == (WEB / path).read_bytes()


@pytest.mark.parametrize(
    ("page", "url"), [("index.html", "/"), ("page.html", "/pages/x")]
)
def test_a_page_loads_its_scripts_and_styles_from_the_app_alone(client, page, url):
    html = (WEB / page).read_text()
    loaded = re.findall(r'<(?:script|link)\b[^>]*\b(?:src|href)="([^"]+)"', html)
    assert any("vendor/" in each for each in loaded)
    for each in loaded:
        assert "//" not in each, each
        response = client.get(urljoin(url, each))
        assert response.status_code == 200, each


def test_the_api_returns_the_library_as_the_engine_resolves_it(client):
    library = client.get("/api/library").json()
    by_name = {each["name"]: each for each in library["namespaces"]}
    settings = {row["label"]: row["items"][0] for row in library["run_settings"]}
    assert library["file"] == "library.sqlite"
    assert library["default_namespace"] == "router"
    assert list(by_name) == [
        "root",
        "course_advisor",
        "health_advisor",
        HISTORIAN,
        "librarian",
        "router",
    ]
    assert [each["name"] for each in by_name["router"]["decompositions"]] == [
        "route a course question",
        "route a wellbeing question",
        "decline anything else",
    ]
    assert field(by_name["root"], "Tools") == [
        {"value": "llm", "source": "granted here", "here": True}
    ]
    assert field(by_name["router"], "Tools")[0]["source"] == "inherited from root"
    assert "model" not in settings and "client" not in settings
    assert settings["max_depth"] == {"value": "8", "source": "the engine's default"}
    assert [(tool["name"], tool["granted_in"]) for tool in library["tools"]] == [
        ("llm", ["root"]),
        ("history", [HISTORIAN]),
        ("library", ["librarian"]),
        ("page", [HISTORIAN]),
    ]
    [catalog] = library["variables"]
    assert (catalog["name"], catalog["where"], len(catalog["rows"])) == (
        "catalog",
        "course_advisor",
        6,
    )


def test_a_fresh_library_shows_llm_granted_at_root_history_and_page_in_the_historian_and_library_in_the_librarian(
    library, home, tmp_path
):
    child = tmp_path / "child.yaml"
    child.write_text("namespaces:\n  helper: {spawn: []}\n")
    with closing(open_library(library)) as connection:
        import_config(connection, child)
    client = TestClient(create_app(library, home), base_url=LOCAL)
    by_name = namespaces(client)
    tools = client.get("/api/library").json()["tools"]
    assert field(by_name["root"], "Tools") == [
        {"value": "llm", "source": "granted here", "here": True},
    ]
    assert field(by_name["helper"], "Tools") == [
        {"value": "llm", "source": "inherited from root", "here": False},
    ]
    assert field(by_name[HISTORIAN], "Tools") == [
        {"value": "llm", "source": "inherited from root", "here": False},
        {"value": "history", "source": "granted here", "here": True},
        {"value": "page", "source": "granted here", "here": True},
    ]
    assert field(by_name[HISTORIAN], "System suffix") == [
        {"value": HISTORIAN_SUFFIX.strip(), "source": "set here", "here": True},
    ]
    assert field(by_name["librarian"], "Tools")[-1] == {
        "value": "library",
        "source": "granted here",
        "here": True,
    }
    assert [(tool["name"], tool["granted_in"]) for tool in tools] == [
        ("llm", ["root"]),
        ("history", [HISTORIAN]),
        ("library", ["librarian"]),
        ("page", [HISTORIAN]),
    ]
    assert [(tool["defined_in"], tool["badge"]) for tool in tools] == [
        (None, "built-in"),
        ("Deep Reasoning", "the app's"),
        ("Deep Reasoning", "the app's"),
        ("Deep Reasoning", "the app's"),
    ]
    assert tools[-1]["description"] == PAGE_DESCRIPTION


def test_a_valid_decomposition_is_stored_and_listed_under_its_namespace(client):
    response = client.post(
        "/api/decompositions", json={"namespace": "course_advisor", "yaml": VALID}
    )
    assert response.status_code == 201
    assert response.json() == {
        "namespace": "course_advisor",
        "name": "next course",
        "yaml": VALID,
        "problems": ["Turn 2 (assistant) has no <think>…</think>."],
    }
    stored = namespaces(client)["course_advisor"]["decompositions"]
    assert [each["name"] for each in stored] == ["catalog lookup", "next course"]


@pytest.mark.parametrize(
    ("namespace", "text", "problem"),
    [
        ("router", "name: x\nmessages: [\n", "The YAML doesn't parse (line 3:"),
        (
            "router",
            VALID + "namespace: router\n",
            'Remove "namespace": only name and messages',
        ),
        (
            "router",
            VALID.replace("role: user", "role: robot"),
            'Message 1: role should be user, assistant or system, not "robot".',
        ),
        ("router", "name: x\n", "Add messages: a list of turns"),
        ("nowhere", VALID, 'The Library has no namespace "nowhere"'),
        (
            "course_advisor",
            VALID.replace("next course", "catalog lookup"),
            'course_advisor already has "catalog lookup"; rename it',
        ),
        (
            "router",
            VALID.replace('"CS310"', '"{{ next }}"'),
            "Turn 2 (assistant), line 1: the engine reads text holding {{ or {%",
        ),
        ("router", VALID.replace("next course", "'next course '"), "starts or ends"),
    ],
)
def test_an_invalid_decomposition_is_refused_with_a_plain_message(
    client, namespace, text, problem
):
    before = namespaces(client)
    response = client.post(
        "/api/decompositions", json={"namespace": namespace, "yaml": text}
    )
    assert response.status_code == 400
    assert problem in response.json()["detail"]
    assert namespaces(client) == before


def test_a_namespace_is_added_edited_and_deleted_through_the_api(client):
    added = client.post(
        "/api/namespaces", json={"name": "research", "yaml": "vars: {topic: ml}\n"}
    )
    child = client.post("/api/namespaces", json={"name": "research.papers", "yaml": ""})
    edited = client.put(
        "/api/namespaces/research.papers",
        json={"yaml": "vars: {year: 2026}\n", "was": ""},
    )
    assert (added.status_code, child.status_code, edited.status_code) == (201, 201, 200)
    papers = by_name(edited.json()["namespaces"])["research.papers"]
    assert (papers["depth"], papers["yaml"]) == (2, "vars: {year: 2026}\n")
    assert field(papers, "Variables") == [
        {"value": "topic (ml)", "source": "inherited from research", "here": False},
        {"value": "year (2026)", "source": "set here", "here": True},
    ]
    refused = client.delete("/api/namespaces/research?decompositions=0")
    assert (refused.status_code, refused.json()) == (
        400,
        {"detail": "Delete its namespaces first: research.papers."},
    )
    client.delete("/api/namespaces/research.papers?decompositions=0")
    deleted = client.delete("/api/namespaces/research?decompositions=0")
    assert deleted.status_code == 200
    assert "research" not in by_name(deleted.json()["namespaces"])
    assert set(namespaces(client)) == {
        "root",
        "router",
        "course_advisor",
        "health_advisor",
        HISTORIAN,
        "librarian",
    }


@pytest.mark.parametrize(
    ("method", "url", "body", "problem"),
    [
        (
            "POST",
            "/api/namespaces",
            {"name": "research", "yaml": "name: research\n"},
            'Remove "name": a namespace is named in the name field, not in its YAML.',
        ),
        (
            "PUT",
            "/api/namespaces/root",
            {"yaml": "tools: [ghost]\n", "was": "tools:\n- llm\n"},
            "The engine refuses it: namespace 'root' grants unknown tool 'ghost'",
        ),
        (
            "DELETE",
            "/api/namespaces/root?decompositions=0",
            None,
            "root can't be deleted",
        ),
        (
            "DELETE",
            "/api/namespaces/router?decompositions=0",
            None,
            "router has 3 decompositions now, not 0",
        ),
    ],
)
def test_a_refused_namespace_change_says_why_and_changes_nothing(
    client, method, url, body, problem
):
    before = client.get("/api/library").json()
    response = client.request(method, url, json=body)
    assert response.status_code == 400
    assert response.json()["detail"].startswith(problem)
    assert client.get("/api/library").json() == before


@pytest.mark.anyio
async def test_a_chat_whose_namespace_was_deleted_is_refused_on_its_next_run(
    library_on, connect
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            steps = [
                await api.post("/api/namespaces", json={"name": "drafts", "yaml": ""}),
                await api.patch(f"/api/chats/{chat}", json={"namespace": "drafts"}),
                await api.delete("/api/namespaces/drafts?decompositions=0"),
            ]
            refused = await api.post(
                f"/api/chats/{chat}/runs", json={"prompt": "Count to two."}
            )
            read = (await api.get(f"/api/chats/{chat}")).json()
    assert [step.status_code for step in steps] == [201, 200, 200]
    assert refused.status_code == 400
    assert refused.json() == {
        "detail": "This chat's namespace drafts no longer exists — pick another."
    }
    assert (read["namespace"], read["turns"]) == ("drafts", [])
    assert fake.calls == []


@pytest.mark.anyio
async def test_a_chat_in_a_namespace_the_engine_cant_read_is_refused_saying_so(
    library_on, connect
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        library = library_on(fake)
        plant_row(library, "namespaces", ("coder", "reasoner: claude_code\n"))
        async with connect(library) as api:
            chat = await new_chat(api)
            await api.patch(f"/api/chats/{chat}", json={"namespace": "coder"})
            refused = await api.post(
                f"/api/chats/{chat}/runs", json={"prompt": "Count to two."}
            )
    assert (refused.status_code, refused.json()) == (
        400,
        {
            "detail": "This chat's namespace coder is one the engine can't read (the"
            " Library panel says why) — pick another."
        },
    )
    assert fake.calls == []


def test_a_namespace_whose_name_holds_a_slash_can_be_edited_and_deleted(
    client, imported
):
    with closing(open_library(imported)) as connection, writing(connection):
        connection.execute(UPSERT_NAMESPACE, ("team/a", ""))
    edited = client.put(
        "/api/namespaces/team%2Fa", json={"yaml": "vars: {x: 1}\n", "was": ""}
    )
    assert by_name(edited.json()["namespaces"])["team/a"]["yaml"] == "vars: {x: 1}\n"
    deleted = client.delete("/api/namespaces/team%2Fa?decompositions=0")
    assert deleted.status_code == 200
    assert "team/a" not in namespaces(client)


def decompositions_of(client, namespace):
    return {
        each["name"]: each for each in namespaces(client)[namespace]["decompositions"]
    }


def test_a_decomposition_is_edited_under_a_new_name_and_deleted_through_the_api(
    client,
):
    [(name, stored), *_] = [
        (each["name"], each["yaml"])
        for each in decompositions_of(client, "root").values()
    ]
    renamed = VALID.replace("next course", "hand it on")
    edited = client.put(
        "/api/decompositions",
        json={"namespace": "root", "name": name, "was": stored, "yaml": renamed},
    )
    assert edited.status_code == 200, edited.text
    shown = by_name(by_name(edited.json()["namespaces"])["root"]["decompositions"])
    assert name not in shown
    assert shown["hand it on"]["problems"] == [
        "Turn 2 (assistant) has no <think>…</think>."
    ]
    deleted = client.request(
        "DELETE",
        "/api/decompositions",
        json={"namespace": "root", "name": "hand it on", "was": renamed},
    )
    assert deleted.status_code == 200, deleted.text
    assert "hand it on" not in decompositions_of(client, "root")


def test_an_edit_the_engine_couldnt_render_is_refused_and_changes_nothing(client):
    [(name, stored), *_] = [
        (each["name"], each["yaml"])
        for each in decompositions_of(client, "root").values()
    ]
    before = client.get("/api/library").json()
    body = {"namespace": "root", "name": name, "was": stored}
    response = client.put(
        "/api/decompositions", json=body | {"yaml": VALID.replace("CS310", "{% x %}")}
    )
    assert response.status_code == 400
    assert (
        "this one doesn't parse (Encountered unknown tag 'x'"
        in response.json()["detail"]
    )
    assert client.get("/api/library").json() == before


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("PUT", "/api/namespaces/router", {"yaml": "", "was": "spawn: [nowhere]\n"}),
        (
            "PUT",
            "/api/decompositions",
            {
                "namespace": "router",
                "name": "decline anything else",
                "was": VALID,
                "yaml": VALID,
            },
        ),
        (
            "DELETE",
            "/api/decompositions",
            {"namespace": "router", "name": "decline anything else", "was": VALID},
        ),
    ],
    ids=["namespace", "decomposition", "delete"],
)
def test_a_save_or_delete_over_a_change_made_meanwhile_is_refused_with_a_conflict(
    client, method, url, body
):
    before = client.get("/api/library").json()
    response = client.request(method, url, json=body)
    assert response.status_code == 409
    assert "was changed after you opened it" in response.json()["detail"]
    assert client.get("/api/library").json() == before


def test_a_tool_granted_nowhere_and_a_run_setting_the_engine_doesnt_read_are_deleted(
    client, imported
):
    block = dump({"factory": "build", "factory_from": "tally.py"})
    with closing(open_library(imported)) as connection, writing(connection):
        connection.execute(UPSERT_TOOL, ("tally", block, "def build(c, p): pass\n"))
        connection.execute(UPSERT_SETTING, ("catalog_path", dump("credits.yaml")))
    tool = client.delete("/api/tools/tally")
    setting = client.delete("/api/run-settings/catalog_path")
    assert (tool.status_code, setting.status_code) == (200, 200)
    assert "tally" not in by_name(setting.json()["tools"])
    assert "catalog_path" not in by_name(setting.json()["run_settings"], "label")


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("/api/tools/nothing", 'The Library holds no tool "nothing" to delete.'),
        ("/api/run-settings/max_iter", "max_iter is a run setting the engine reads"),
    ],
)
def test_a_tool_or_run_setting_that_cant_be_deleted_is_refused_in_words(
    client, url, problem
):
    before = client.get("/api/library").json()
    response = client.delete(url)
    assert response.status_code == 400
    assert response.json()["detail"].startswith(problem)
    assert client.get("/api/library").json() == before


def test_a_granted_tool_is_not_deleted_and_the_refusal_names_who_grants_it(
    client, imported
):
    block = dump({"factory": "build", "factory_from": "tally.py"})
    with closing(open_library(imported)) as connection, writing(connection):
        connection.execute(UPSERT_TOOL, ("tally", block, "def build(c, p): pass\n"))
    client.post(
        "/api/namespaces", json={"name": "counting", "yaml": "tools: [tally]\n"}
    )
    response = client.delete("/api/tools/tally")
    assert (response.status_code, response.json()) == (
        400,
        {"detail": "tally is granted in counting: take it out of their tools first."},
    )


def test_a_request_naming_another_host_is_refused(client):
    response = client.get("/api/library", headers={"Host": "rebound.example"})
    assert response.status_code == 400


@pytest.mark.anyio
async def test_a_message_runs_and_its_events_stream_with_ids(library_on, connect):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            run = await send(api, await new_chat(api), "Count to two.")
            response = await api.get(f"/api/runs/{run}/events")
            resumed = await api.get(
                f"/api/runs/{run}/events", headers={"Last-Event-ID": "3"}
            )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = server_sent(response.text)
    assert [seq for seq, _ in events] == list(range(1, len(events) + 1))
    assert all(event["seq"] == seq for seq, event in events)
    assert events[0][1]["kind"] == "turn.start" and events[0][1]["namespace"] == "root"
    assert events[-1][1]["outcome"] == "answered" and events[-1][1]["answer"] == "2"
    assert server_sent(resumed.text) == events[3:]


@pytest.mark.anyio
async def test_a_stream_resumes_after_its_query_or_its_header_whichever_is_later(
    library_on, connect
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            run = await send(api, await new_chat(api), "Count to two.")
            url = f"/api/runs/{run}/events"
            events = server_sent((await api.get(url)).text)
            after = await api.get(url, params={"after": 3})
            reconnected = await api.get(
                url, params={"after": 3}, headers={"Last-Event-ID": "5"}
            )
    assert server_sent(after.text) == events[3:]
    assert server_sent(reconnected.text) == events[5:]


@pytest.mark.anyio
async def test_a_follow_up_gets_the_chat_so_far_as_turns_after_the_examples(
    library_on, connect
):
    plan = PLAN | {
        "Count to two.": [repl("print(1)"), repl("FinalAnswer(2)", think="Then 2.")],
        "And what did you answer?": [repl("FinalAnswer('2')")],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat, another = await new_chat(api), await new_chat(api)
            await run_events(api, await send(api, chat, "Count to two."))
            await run_events(api, await send(api, chat, "What did I ask?"))
            asked = len(fake.calls)
            await run_events(api, await send(api, chat, "And what did you answer?"))
            await run_events(api, await send(api, another, "What did I ask?"))
    third, fresh = fake.calls[asked].messages, fake.calls[-1].messages
    assert [(message["role"], message["content"]) for message in third[-7:]] == [
        ("system", DEFAULT_EXAMPLE_FENCE),
        ("system", EARLIER_TURNS_NOTE),
        ("user", "Count to two."),
        ("assistant", folded_answer("'2'")),
        ("user", "What did I ask?"),
        ("assistant", folded_answer("'to count'")),
        ("user", "And what did you answer?"),
    ]
    assert [(message["role"], message["content"]) for message in fresh[-2:]] == [
        ("system", DEFAULT_EXAMPLE_FENCE),
        ("user", "What did I ask?"),
    ]


@pytest.mark.anyio
async def test_a_chat_keeps_the_namespace_it_started_in(imported, library_on, connect):
    plan = {"Hi.": [repl("FinalAnswer('hello')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat, another = await new_chat(api), await new_chat(api)
            first = await run_events(api, await send(api, chat, "Hi."))
            with closing(open_library(imported)) as connection, writing(connection):
                default = ("entry_namespace", dump("course_advisor"))
                connection.execute(UPSERT_SETTING, default)
            again = await run_events(api, await send(api, chat, "Hi."))
            new = await run_events(api, await send(api, another, "Hi."))
    namespaces = [events[0]["namespace"] for events in (first, again, new)]
    assert namespaces == ["router", "router", "course_advisor"]
    assert [events[-1]["outcome"] for events in (first, again, new)] == ["answered"] * 3


@pytest.mark.anyio
async def test_a_run_gets_the_librarys_tools_in_a_folder_of_its_own(
    waitlist, library, library_on, connect, data
):
    with closing(open_library(library)) as connection:
        import_config(connection, waitlist)
    plan = {"Join.": [repl("FinalAnswer(waitlist.join('ada', 'CS101'))")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            run = await send(api, await new_chat(api), "Join.")
            events = await run_events(api, run)
    assert events[-1]["outcome"] == "answered", events[-1]
    assert events[-1]["answer"] == "enrolled"
    assert (data / "runs" / run / "tools" / "waitlist.py").is_file()


@pytest.mark.anyio
async def test_a_run_on_a_fresh_library_gets_the_llm_tools_answer(library_on, connect):
    plan = {
        "Ask the llm tool.": [repl("FinalAnswer(llm('Name a prime.'))")],
        "Name a prime.": ["7"],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            run = await send(api, await new_chat(api), "Ask the llm tool.")
            events = await run_events(api, run)
    assert events[-1]["outcome"] == "answered", events[-1]
    assert events[-1]["answer"] == "7"


# Templates a shared file may hold that read a Python internal: each renders a name
# unsandboxed, and the sandbox refuses it.
PROBES = {
    "through a jinja global": "{{ cycler.__init__.__name__ }}",
    "through the task alone": "{{ task.__class__.__name__ }}",
}
PROBE_PLAN = {"what is 2+2?": [repl("FinalAnswer(4)")]}


def stored(library, upsert, *row):
    """A row written straight into the app file, as a shared file may hold it."""
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(upsert, row)


async def ask_plainly(library, connect):
    """A plain message in a new chat of *library*: its run's events."""
    async with connect(library) as api:
        return await run_events(
            api, await send(api, await new_chat(api), "hello, what is 2+2?")
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("template", "why"),
    [
        (PROBES["through a jinja global"], "access to attribute '__init__' of 'type'"),
        (PROBES["through the task alone"], "access to attribute '__class__' of 'str'"),
        ("{% include 'x.html' %}", "x.html"),
    ],
    ids=[*PROBES, "including another template"],
)
async def test_a_shared_files_worked_example_no_run_renders_fails_the_run_naming_it(
    template, why, library_on, connect
):
    example = {
        "name": "innocent",
        "messages": [
            {"role": "user", "content": f"Example: {template}"},
            {"role": "assistant", "content": repl("FinalAnswer(1)", think="ok")},
        ],
    }
    with FakeOpenAI(scripted(PROBE_PLAN)) as fake:
        library = library_on(fake)
        stored(library, UPSERT_DECOMPOSITION, "root", "innocent", dump(example))
        events = await ask_plainly(library, connect)
    end = events[-1]
    assert end["outcome"] == "failed"
    assert end["detail"].startswith(
        "ValueError: decomposition 'innocent' in namespace 'root' could not be"
        f" rendered: {why}"
    )
    assert fake.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("probe", PROBES.values(), ids=PROBES)
@pytest.mark.parametrize("where", ["system_suffix", "system_prompt"])
async def test_a_shared_files_prompt_reading_python_internals_is_shown_as_written(
    probe, where, library_on, connect
):
    text = f"Answer for {probe}."
    with FakeOpenAI(scripted(PROBE_PLAN)) as fake:
        library = library_on(fake)
        if where == "system_suffix":
            root = {"tools": ["llm"], "system_suffix": text}
            stored(library, UPSERT_NAMESPACE, "root", dump(root))
        else:
            stored(library, UPSERT_SETTING, "system_prompt", dump(text))
        events = await ask_plainly(library, connect)
    assert events[-1]["outcome"] == "answered", events[-1]
    assert any(text in each["content"] for each in fake.calls[0].messages)


@pytest.mark.anyio
async def test_a_chats_earlier_messages_and_answers_are_never_rendered(
    library_on, connect
):
    asked = f"Echo {{{{ 6 * 7 }}}} {PROBES['through a jinja global']}"
    answer = f"{{{{ 7 * 6 }}}} {PROBES['through the task alone']}"
    plan = {
        "Echo": [repl(f"FinalAnswer({answer!r})")],
        "And now?": [repl("FinalAnswer(0)")],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            first = await run_events(api, await send(api, chat, asked))
            then = await run_events(api, await send(api, chat, "And now?"))
    assert [first[-1]["answer"], then[-1]["outcome"]] == [answer, "answered"]
    shown = [(each["role"], each["content"]) for each in fake.calls[-1].messages]
    assert shown[-3:] == [
        ("user", asked),
        ("assistant", folded_answer(repr(answer))),
        ("user", "And now?"),
    ]


def historian_or(historian, others):
    """A responder that asks *historian* for the historian, whose system prompt holds its
    suffix, and *others* for every other agent: the historian's task is the user's
    message, as root's is."""

    def respond(messages):
        system = [each["content"] for each in messages if each["role"] == "system"]
        told = any(HISTORIAN_SUFFIX.strip() in each for each in system)
        return (historian if told else others)(messages)

    return respond


ASKED_HOW = "How did you count?"
# Root's hand-off, as its worked example writes it.
HANDING_OFF = parsed(hand_off(HISTORIANS_EXAMPLES[0])).messages[-1].content
HISTORY_PLAN = {
    "Count to two.": [repl("print(1)"), repl("FinalAnswer(2)")],
    ASKED_HOW: [HANDING_OFF],
}
HISTORIANS_PLAN = {
    ASKED_HOW: [
        repl("t = history.turns()", "print(len(t), t[0]['message'], t[0]['answer'])"),
        repl("print(history.steps(1))"),
        repl("print(history.steps(9))"),
        repl("FinalAnswer('by printing 1')"),
    ]
}


@pytest.mark.anyio
async def test_root_hands_a_question_about_an_earlier_message_to_the_historian(
    library_on, connect
):
    responder = historian_or(scripted(HISTORIANS_PLAN), scripted(HISTORY_PLAN))
    with FakeOpenAI(responder) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            await run_events(api, await send(api, chat, "Count to two."))
            asked = len(fake.calls)
            events = await run_events(api, await send(api, chat, ASKED_HOW))
    root, historian = kinds(events, "agent.start")
    assert (historian["namespace"], historian["parent"]) == (HISTORIAN, root["node"])
    assert historian["task"] == ASKED_HOW
    turns, steps, refused, _, handed_off = kinds(events, "cell.end")
    assert handed_off["code"] == running_code(HANDING_OFF)
    assert turns["output"] == "1 Count to two. 2"
    assert "'code': 'print(1)'" in steps["output"]
    assert "'code': 'FinalAnswer(2)'" in steps["output"]
    assert refused["status"] == "failed"
    assert "ValueError: There is no message 9" in refused["output"]
    assert events[-1]["outcome"] == "answered", events[-1]
    assert events[-1]["answer"] == "by printing 1"
    roots_prompt, historians_prompt = [
        "\n".join(each["content"] for each in call.messages if each["role"] == "system")
        for call in (fake.calls[asked], fake.calls[-1])
    ]
    assert HISTORY_DESCRIPTION not in roots_prompt
    assert HISTORY_DESCRIPTION in historians_prompt
    assert HISTORIAN_SUFFIX.strip() in historians_prompt


@pytest.mark.anyio
async def test_a_historian_that_copies_roots_hand_off_nests_no_historian(
    library_on, connect
):
    copying = {ASKED_HOW: [HANDING_OFF, repl("FinalAnswer('by printing 1')")]}
    responder = historian_or(scripted(copying), scripted(HISTORY_PLAN))
    with FakeOpenAI(responder) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            await run_events(api, await send(api, chat, "Count to two."))
            events = await run_events(api, await send(api, chat, ASKED_HOW))
    namespaces = [each["namespace"] for each in kinds(events, "agent.start")]
    assert namespaces == ["root", HISTORIAN]
    copied = kinds(events, "cell.end")[0]
    assert copied["status"] == "failed"
    assert "may not spawn a sub-agent in 'historian'" in copied["output"]
    assert events[-1]["answer"] == "by printing 1"


def copying_its_first_example(others):
    """A model that answers ASKED_HOW by copying its first worked example's reply, as models
    do, and answers the rest as *others* does."""
    question = parsed(HISTORIANS_EXAMPLES[0]).messages[0].content

    def respond(messages):
        if messages[-1]["content"] != ASKED_HOW:
            return others(messages)
        for asked, reply in itertools.pairwise(messages):
            if asked["content"] == question and reply["role"] == "assistant":
                return reply["content"]
        raise AssertionError("no worked example asks the question")

    return respond


@pytest.mark.anyio
@pytest.mark.parametrize("spawn", ["spawn: [trips]\n", "spawn: []\n"])
async def test_a_namespace_that_may_not_hand_off_to_the_historian_says_it_cant(
    library_on, connect, spawn
):
    # Copying root's hand-off, its agent was refused, then handed the question to another
    # namespace, which inherits the same example, until the depth ran out.
    plan = {"Count to two.": [repl("print(1)"), repl("FinalAnswer(2)")]}
    with FakeOpenAI(copying_its_first_example(scripted(plan))) as fake:
        async with connect(library_on(fake)) as api:
            added = await api.post(
                "/api/namespaces", json={"name": "trips", "yaml": spawn}
            )
            assert added.status_code == 201, added.text
            chat = await chat_in(api, "trips")
            await run_events(api, await send(api, chat, "Count to two."))
            events = await run_events(api, await send(api, chat, ASKED_HOW))
    assert [each["namespace"] for each in kinds(events, "agent.start")] == ["trips"]
    [cell] = kinds(events, "cell.end")
    assert cell["status"] == "done"
    assert events[-1]["answer"] == (
        "I can't answer that here: questions like it go to the historian namespace,"
        " and this namespace's spawn list leaves it out."
    )


# Message 1 of the examples that read sub-agents, as the fake answers it.
SUB_AGENTS_FIRST_MESSAGE = (
    "Add the sum of the first 10 primes to the 10th Fibonacci number.",
    {
        "Add the sum": [
            repl(
                'primes = subagent("Sum the first 10 primes.")',
                'fib = subagent("Find the 10th Fibonacci number.")',
                "print(primes, fib)",
                think="Give each part to its own sub-agent, then add the two.",
            ),
            repl("FinalAnswer(primes + fib)"),
        ],
        "Sum the first 10 primes.": [
            repl(
                "primes = [n for n in range(2, 30) if all(n % d for d in range(2, n))]",
                "FinalAnswer(sum(primes))",
                think="The primes below 30 are the first 10; sum them.",
            )
        ],
        "Find the 10th Fibonacci number.": [
            repl(
                "a, b = 0, 1",
                "for _ in range(10):",
                "    a, b = b, a + b",
                "FinalAnswer(a)",
                think="Step the pair ten times.",
            )
        ],
    },
)
# Message 1 of the example that finds an error where the user didn't say: the top agent's
# cell that started the sub-agents fails on what one of them answered, and the next one
# recovers.
FAILED_IN_THE_TOP_AGENT = (
    "How many primes are below 20,000?",
    {
        "How many primes are below 20,000?": [
            repl(
                "parts = [subagent(f'Count the primes in [{a}, {a + 10_000}).')"
                " for a in (0, 10_000)]",
                "print(parts)",
                "print(sum(int(part) for part in parts))",
                think="Split the range between two sub-agents, then add their counts.",
            ),
            repl(
                "total = sum(int(str(part).replace(',', '')) for part in parts)",
                "FinalAnswer(total)",
            ),
        ],
        "Count the primes in [0, 10000).": [
            repl(
                "FinalAnswer(sum(all(n % d for d in range(2, int(n**0.5) + 1))"
                " for n in range(2, 10_000)))"
            )
        ],
        "Count the primes in [10000, 20000).": [
            repl(
                "count = sum(all(n % d for d in range(2, int(n**0.5) + 1))"
                " for n in range(10_000, 20_000))",
                "FinalAnswer(f'{count:,}')",
            )
        ],
    },
)
# The first message each of the historian's examples explains, as the fake answers it.
EXAMPLES_FIRST_MESSAGE = {
    "explain an earlier answer": (
        "What is 17 times 23?",
        {"What is 17 times 23?": [repl("FinalAnswer(17 * 23)")]},
    ),
    # The answer typed in by hand, and wrong: pow gives 184.
    "tell a typed answer from a computed one": (
        "What is 13**31 mod 1009?",
        {"What is 13**31 mod 1009?": [repl("pow(13, 31, 1009)", "FinalAnswer(512)")]},
    ),
    "say what each sub-agent did": SUB_AGENTS_FIRST_MESSAGE,
    "show it on a page": SUB_AGENTS_FIRST_MESSAGE,
    "put a table on a page": SUB_AGENTS_FIRST_MESSAGE,
    "find where it really went wrong": FAILED_IN_THE_TOP_AGENT,
    "say when there is nothing to show": SUB_AGENTS_FIRST_MESSAGE,
}
# A cost depends on the prompt, which the examples are part of, and a time on the run.
VARIES = re.compile(r"'(cost|seconds)': [^,}]+")


def as_recorded(observation):
    """*observation* as any run of an example prints it: its costs and times masked."""
    return VARIES.sub(r"'\1': …", observation)


# How a hand-off reaches the historian: the question, root's hand-off, then the question
# again as its task.
HANDED_OVER_TURNS = HAND_OFF_TURNS + 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text", HISTORIANS_EXAMPLES, ids=lambda text: parsed(text).name
)
async def test_the_historians_examples_show_what_history_really_prints(
    library_on, connect, text
):
    example = parsed(text)
    opening = example.messages[:HANDED_OVER_TURNS]
    rest = example.messages[HANDED_OVER_TURNS:]
    fake, asked, events, _, _ = await ask_as_the_example(library_on, connect, text)
    # The historian's conversation ends as the example opens.
    historians_first = fake.calls[asked + 1].messages[-HANDED_OVER_TURNS:]
    assert [(each["role"], each["content"].strip()) for each in historians_first] == [
        (each.role, each.content.strip()) for each in opening
    ]
    shown = [turn.content for turn in rest if turn.role == "user"]
    printed = [
        f"<observation>\n{cell['output']}\n</observation>\n"
        for cell in kinds(events, "cell.end")[: len(shown)]
    ]
    assert list(map(as_recorded, printed)) == list(map(as_recorded, shown))
    assert events[-1]["outcome"] == "answered", events[-1]


async def ask_as_the_example(library_on, connect, text):
    """The historian's example *text* run through the app on the fake: its first message,
    then its question, which root hands off and the historian answers with the example's
    own cells. The fake, how many calls it had before the question, the question's
    events, the chat's turns, and each page opened as the API serves it."""
    example = parsed(text)
    question, handing_off, _ = example.messages[:HANDED_OVER_TURNS]
    rest = example.messages[HANDED_OVER_TURNS:]
    first, plan = EXAMPLES_FIRST_MESSAGE[example.name]
    replies = [turn.content for turn in rest if turn.role == "assistant"]
    plan = plan | {question.content: [handing_off.content]}
    historian = scripted({question.content: replies})
    with FakeOpenAI(historian_or(historian, scripted(plan))) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            await run_events(api, await send(api, chat, first))
            asked = len(fake.calls)
            events = await run_events(api, await send(api, chat, question.content))
            turns = (await api.get(f"/api/chats/{chat}")).json()["turns"]
            pages = [
                (await api.get(f"/api/pages/{each['page']}")).json()
                for each in kinds(events, "page.open")
            ]
    return fake, asked, events, turns, pages


@pytest.mark.anyio
async def test_root_hands_show_me_on_a_page_to_the_historian_whose_page_its_turn_links(
    library_on, connect
):
    [text] = [
        each for each in HISTORIANS_EXAMPLES if parsed(each).name == "show it on a page"
    ]
    _, _, events, turns, pages = await ask_as_the_example(library_on, connect, text)
    _, historian = kinds(events, "agent.start")
    [opened] = kinds(events, "page.open")
    [page] = pages
    assert historian["namespace"] == HISTORIAN
    assert (opened["node"], opened["title"]) == (
        historian["node"],
        "How the answer to message 1 was reached",
    )
    assert kinds(turns[1]["events"], "page.open") == [opened]
    assert events[-1]["answer"].startswith("The answer, 184, came in two parts")
    first = turns[0]["events"]
    first_root = kinds(first, "agent.start")[0]
    delegating, adding = [
        each["code"]
        for each in kinds(first, "cell.end")
        if each["node"] == first_root["node"]
    ]
    text_block, fanned_out, sub_agent, added = page["blocks"]
    assert text_block["type"] == "text"
    assert [
        (each["turn"], each["node"], each["index"]) for each in (fanned_out, added)
    ] == [
        (1, None, 1),
        (1, None, 2),
    ]
    assert (fanned_out["step"]["code"], added["step"]["code"]) == (delegating, adding)
    assert [row["task"] for row in fanned_out["step"]["agents"]] == [
        "Sum the first 10 primes.",
        "Find the 10th Fibonacci number.",
    ]
    [drive] = sub_agent["agents"]
    assert (sub_agent["node"], drive["task"], drive["result"]) == (
        2,
        "Sum the first 10 primes.",
        "129",
    )


@pytest.mark.anyio
async def test_the_historians_table_on_a_page_holds_the_sub_agents_as_recorded(
    library_on, connect
):
    [text] = [
        each
        for each in HISTORIANS_EXAMPLES
        if parsed(each).name == "put a table on a page"
    ]
    _, _, events, turns, pages = await ask_as_the_example(library_on, connect, text)
    [[table]] = [page["blocks"] for page in pages]
    sub_agents = [
        each for each in kinds(turns[0]["events"], "agent.start") if each["parent"]
    ]
    answers = {
        each["node"]: each["result"] for each in kinds(turns[0]["events"], "agent.end")
    }
    assert table["columns"] == ["asked", "answer", "cost"]
    assert [row[:2] for row in table["rows"]] == [
        [each["task"], answers[each["node"]]] for each in sub_agents
    ]
    assert events[-1]["answer"].startswith("Message 1 had two sub-agents")


@pytest.mark.anyio
async def test_the_historian_shows_an_error_put_in_a_sub_agent_where_it_really_was(
    library_on, connect
):
    [text] = [
        each
        for each in HISTORIANS_EXAMPLES
        if parsed(each).name == "find where it really went wrong"
    ]
    _, _, events, turns, pages = await ask_as_the_example(library_on, connect, text)
    first = turns[0]["events"]
    top = kinds(first, "agent.start")[0]
    [failed] = [each for each in kinds(first, "cell.end") if each["status"] == "failed"]
    [[block]] = [page["blocks"] for page in pages]
    assert failed["node"] == top["node"]
    assert (block["node"], block["step"]["code"], block["step"]["status"]) == (
        None,
        failed["code"],
        "failed",
    )
    assert [row["status"] for row in block["step"]["agents"]] == ["done", "done"]
    assert events[-1]["answer"].startswith("No sub-agent had an error")


# What EXAMPLE and REWORDED lack as worked examples: their one reply is prose.
EXAMPLE_PROBLEMS = [
    "Turn 2 (assistant) has no <think>…</think>.",
    "Turn 2 (assistant) has no <repl>…</repl> block.",
    "Turn 2 (assistant), the last, should call FinalAnswer(...) in its <repl>.",
]
LIBRARY_PLAN = {
    "Save the example.": [write("librarian", EXAMPLE), repl("FinalAnswer('saved')")],
    "Reword it.": [write("librarian", REWORDED), repl("FinalAnswer('reworded')")],
    "Thanks.": [repl("FinalAnswer('welcome')")],
}
REFUSALS = [
    (write("librarian", "name: [unclosed"), "The YAML doesn't parse ("),
    (write("librarian", "name: count\n"), "Add messages: a list of turns"),
    (write("nowhere", EXAMPLE), 'The Library has no namespace "nowhere"'),
    (write("root", EXAMPLE), "root is locked"),
    (write("librarian", f"{EXAMPLE}# {SERVER_KEY}\n"), "holds a provider's key"),
]


async def chat_in(api, namespace):
    """A new chat in *namespace*."""
    chat = await new_chat(api)
    response = await api.patch(f"/api/chats/{chat}", json={"namespace": namespace})
    assert response.status_code == 200, response.text
    return chat


async def librarian_chat(api):
    """A new chat in the librarian namespace, where the library tool is granted."""
    return await chat_in(api, "librarian")


def kinds(events, kind):
    return [event for event in events if event["kind"] == kind]


def changes(events):
    return kinds(events, "library.change")


@pytest.mark.anyio
async def test_the_library_tool_adds_a_decomposition_then_replaces_it_each_change_a_run_event(
    library_on, connect
):
    with FakeOpenAI(scripted(LIBRARY_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await librarian_chat(api)
            added = await run_events(api, await send(api, chat, "Save the example."))
            replaced = await run_events(api, await send(api, chat, "Reword it."))
            turns = (await api.get(f"/api/chats/{chat}")).json()["turns"]
            namespaces = (await api.get("/api/library")).json()["namespaces"]
    [root] = kinds(added, "agent.start")
    [first], [second] = changes(added), changes(replaced)
    assert {key: first[key] for key in ("node", *CHANGE_FIELDS)} == {
        "node": root["node"],
        "what": "decomposition",
        "namespace": "librarian",
        "name": "count the tickets",
        "before": None,
        "after": EXAMPLE,
    }
    assert (second["name"], second["before"], second["after"]) == (
        "count the tickets",
        EXAMPLE,
        REWORDED,
    )
    assert "'done': 'added'" in kinds(added, "cell.end")[0]["output"]
    assert "'done': 'replaced'" in kinds(replaced, "cell.end")[0]["output"]
    assert [changes(turn["events"]) for turn in turns] == [[first], [second]]
    librarian = next(each for each in namespaces if each["name"] == "librarian")
    output = kinds(replaced, "cell.end")[0]["output"]
    assert f"'problems': {EXAMPLE_PROBLEMS}" in output
    assert librarian["decompositions"] == [
        *LIBRARIANS_SEEDED,
        {"name": "count the tickets", "yaml": REWORDED, "problems": EXAMPLE_PROBLEMS},
    ]
    assert [events[-1]["outcome"] for events in (added, replaced)] == ["answered"] * 2
    assert any("The `library` tool" in each for each in system_prompts(fake.calls[-1:]))


def showing(calls, turn):
    """Whether each of *calls* showed the model *turn* as a message of its own, as an
    example's turn is shown."""
    return [
        any(str(message.get("content")).strip() == turn for message in call.messages)
        for call in calls
    ]


@pytest.mark.anyio
async def test_a_change_to_the_library_applies_from_the_next_message(
    library_on, connect
):
    with FakeOpenAI(scripted(LIBRARY_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await librarian_chat(api)
            calls = []
            for prompt in ("Save the example.", "Reword it.", "Thanks."):
                await run_events(api, await send(api, chat, prompt))
                calls.append(fake.calls[sum(map(len, calls)) :])
    saving, rewording, thanking = calls
    assert showing(saving, ASKED) == [False, False]
    assert showing(rewording, ASKED) == [True, True]
    assert showing(rewording, REWORDED_ASK) == [False, False]
    assert showing(thanking, REWORDED_ASK) == [True]
    assert showing(thanking, ASKED) == [False]


# A fresh Library's decompositions, as the panel lists them: root's hand-offs, the
# historian's examples, and the librarian's.
FRESH_DECOMPOSITIONS = [
    [{"name": parsed(text).name, "yaml": text, "problems": []} for text in texts]
    for texts in (
        [hand_off(text) for text in HISTORIANS_EXAMPLES],
        HISTORIANS_EXAMPLES,
        LIBRARIANS_EXAMPLES,
    )
]
LIBRARIANS_SEEDED = FRESH_DECOMPOSITIONS[-1]


@pytest.mark.anyio
async def test_a_refused_write_fails_its_cell_in_the_panels_words_and_stores_nothing(
    library_on, connect
):
    cells = [cell for cell, _ in REFUSALS]
    plan = {"Try these.": [*cells, repl("FinalAnswer('refused')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await librarian_chat(api)
            events = await run_events(api, await send(api, chat, "Try these."))
            namespaces = (await api.get("/api/library")).json()["namespaces"]
    ended = kinds(events, "cell.end")[: len(REFUSALS)]
    assert [cell["status"] for cell in ended] == ["failed"] * len(REFUSALS)
    for cell, (_, wording) in zip(ended, REFUSALS, strict=True):
        assert wording in cell["output"], cell["output"]
    assert changes(events) == []
    assert [each["decompositions"] for each in namespaces] == FRESH_DECOMPOSITIONS
    assert events[-1]["outcome"] == "answered", events[-1]


NAMESPACE_PLAN = {
    "Add research.": [
        library_cell("add_namespace", "research", RESEARCH),
        repl("FinalAnswer('added')"),
    ],
    "Change research.": [
        library_cell("change_namespace", "research", RESEARCH_CHANGED),
        repl("FinalAnswer('changed')"),
    ],
    "Use research.": [repl("print(year)"), repl("FinalAnswer('used')")],
}


def system_prompts(calls):
    """What each of *calls* told the model as system messages: the prompt, then the
    namespace's suffix."""
    return [
        "\n".join(
            str(message["content"])
            for message in call.messages
            if message.get("role") == "system"
        )
        for call in calls
    ]


@pytest.mark.anyio
async def test_the_library_tool_adds_a_namespace_whose_suffix_and_vars_the_next_run_there_gets(
    library_on, connect
):
    with FakeOpenAI(scripted(NAMESPACE_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            librarian = await librarian_chat(api)
            added = await run_events(api, await send(api, librarian, "Add research."))
            namespaces = (await api.get("/api/library")).json()["namespaces"]
            chat = await chat_in(api, "research")
            asked = len(fake.calls)
            used = await run_events(api, await send(api, chat, "Use research."))
    [change] = changes(added)
    assert {key: change[key] for key in CHANGE_FIELDS} == {
        "what": "namespace",
        "namespace": "research",
        "name": "research",
        "before": None,
        "after": RESEARCH,
    }
    assert "'done': 'added'" in kinds(added, "cell.end")[0]["output"]
    research = next(each for each in namespaces if each["name"] == "research")
    assert research["yaml"] == RESEARCH
    assert kinds(used, "cell.end")[0]["output"] == "2026"
    prompts = system_prompts(fake.calls[asked:])
    assert prompts and all(SUFFIX in prompt for prompt in prompts)
    assert not any(SUFFIX in prompt for prompt in system_prompts(fake.calls[:asked]))


@pytest.mark.anyio
async def test_a_namespace_the_library_tool_changes_has_its_old_yaml_as_before_and_the_next_run_gets_the_new(
    library_on, connect
):
    with FakeOpenAI(scripted(NAMESPACE_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            librarian = await librarian_chat(api)
            await run_events(api, await send(api, librarian, "Add research."))
            changed = await run_events(
                api, await send(api, librarian, "Change research.")
            )
            asked = len(fake.calls)
            chat = await chat_in(api, "research")
            await run_events(api, await send(api, chat, "Use research."))
    [change] = changes(changed)
    assert (change["what"], change["namespace"]) == ("namespace", "research")
    assert (change["before"], change["after"]) == (RESEARCH, RESEARCH_CHANGED)
    assert "'done': 'changed'" in kinds(changed, "cell.end")[0]["output"]
    prompts = system_prompts(fake.calls[asked:])
    assert prompts and all(CHANGED_SUFFIX in prompt for prompt in prompts)
    assert not any(SUFFIX in prompt for prompt in prompts)


@pytest.mark.anyio
async def test_a_namespace_change_event_holds_the_yaml_as_stored_and_none_is_logged_for_no_change(
    library_on, connect
):
    plan = {
        "Add a blank one.": [
            library_cell("add_namespace", "blank", "{}\n"),
            library_cell("change_namespace", "blank", "{}\n"),
            library_cell("change_namespace", "blank", {"system_suffix": "Hi."}),
            repl("FinalAnswer('added')"),
        ]
    }
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await librarian_chat(api)
            events = await run_events(api, await send(api, chat, "Add a blank one."))
    added, changed = changes(events)
    assert (added["before"], added["after"]) == (None, "")
    assert "'done': 'unchanged'" in kinds(events, "cell.end")[1]["output"]
    assert (changed["before"], changed["after"]) == ("", "system_suffix: Hi.\n")


# Set by the user in the panel: what the library tool may neither change nor drop.
DRAFTS = "tools: [history]\nrepl: {type: local}\n"
NAMESPACE_REFUSALS = [
    (library_cell("add_namespace", "root", ""), "root is locked"),
    (
        library_cell("change_namespace", "root", "system_suffix: Hi.\n"),
        "root is locked",
    ),
    (
        library_cell("add_namespace", "research", "repl: {type: local}\n"),
        "repl may not be set or changed: a namespace's repl and reasoner are the user's",
    ),
    (
        library_cell("add_namespace", "research", "reasoner: {type: claude_code}\n"),
        "reasoner may not be set or changed",
    ),
    (
        repl("library.change_namespace('drafts', 'tools: [history]', replace=True)"),
        "repl may not be set or changed",
    ),
    (
        library_cell("add_namespace", "research", "tools: [library]\n"),
        "tools may not add or take out library: which namespaces grant the app's own",
    ),
    (
        library_cell("add_namespace", "research", "tools: [llm, history]\n"),
        "tools may not add or take out history",
    ),
    (
        library_cell("change_namespace", "librarian", "tools: []\n"),
        "tools may not add or take out library",
    ),
    (
        library_cell("change_namespace", "librarian", "tools: [library, history]\n"),
        "tools may not add or take out history",
    ),
    (
        repl("library.change_namespace('drafts', 'repl: {type: local}', replace=True)"),
        "tools may not add or take out history",
    ),
    (
        library_cell("change_namespace", "drafts", {"tools": None}),
        "tools may not add or take out history",
    ),
    (
        library_cell("change_namespace", "historian", "tools: [history]\nspawn: []\n"),
        "tools may not add or take out page",
    ),
    (
        library_cell("add_namespace", "re search", ""),
        "A namespace name is words of letters, digits, _ and -, joined by dots",
    ),
    (
        library_cell("add_namespace", "research.papers", ""),
        'Add namespace "research" first: "research.papers" goes under it.',
    ),
    (
        library_cell("add_namespace", "librarian.notes", ""),
        "librarian.notes would inherit library from librarian: which namespaces grant",
    ),
    (
        library_cell("add_namespace", "research", "spawn: [nowhere]\n"),
        (
            'spawn names "nowhere", which is not a namespace in the Library: add'
            " nowhere first, or take it out of spawn."
        ),
    ),
    (
        library_cell("add_namespace", "research", "tools: [no_such_tool]\n"),
        "The engine refuses it: namespace 'research' grants unknown tool 'no_such_tool'",
    ),
    (
        library_cell("add_namespace", "research", f"system_suffix: {SERVER_KEY}\n"),
        "holds a provider's key",
    ),
    (
        library_cell("change_namespace", "nowhere", ""),
        'The Library has no namespace "nowhere"',
    ),
]


@pytest.mark.anyio
async def test_a_refused_namespace_write_fails_its_cell_in_plain_words_and_changes_nothing(
    library_on, connect
):
    cells = [cell for cell, _ in NAMESPACE_REFUSALS]
    plan = {"Try these.": [*cells, repl("FinalAnswer('refused')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            drafts = {"name": "drafts", "yaml": DRAFTS}
            response = await api.post("/api/namespaces", json=drafts)
            assert response.status_code == 201, response.text
            before = (await api.get("/api/library")).json()
            chat = await librarian_chat(api)
            events = await run_events(api, await send(api, chat, "Try these."))
            after = (await api.get("/api/library")).json()
    ended = kinds(events, "cell.end")[: len(NAMESPACE_REFUSALS)]
    assert [cell["status"] for cell in ended] == ["failed"] * len(NAMESPACE_REFUSALS)
    for cell, (_, wording) in zip(ended, NAMESPACE_REFUSALS, strict=True):
        assert wording in cell["output"], cell["output"]
    assert changes(events) == []
    assert after == before
    assert events[-1]["outcome"] == "answered", events[-1]


TOOL_PLAN = {
    "Make a word counter.": [
        library_cell("write_tool", "word_count", WORD_COUNT, "build"),
        library_cell("change_namespace", "counting", "tools: [word_count]\n"),
        repl("FinalAnswer('made')"),
    ],
    "Split on commas.": [
        library_cell("write_tool", "word_count", SPLIT_COUNT, "build", {"sep": ","}),
        repl("FinalAnswer('split')"),
    ],
    "Count a,b,c d.": [
        repl("print(word_count('a,b,c d'))"),
        repl("FinalAnswer('counted')"),
    ],
}


@pytest.mark.anyio
async def test_a_tool_the_library_tool_writes_and_grants_is_called_from_the_next_message(
    library_on, connect
):
    with FakeOpenAI(scripted(TOOL_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            counting = {"name": "counting", "yaml": ""}
            response = await api.post("/api/namespaces", json=counting)
            assert response.status_code == 201, response.text
            librarian = await librarian_chat(api)
            chat = await chat_in(api, "counting")
            made = await run_events(
                api, await send(api, librarian, "Make a word counter.")
            )
            asked = len(fake.calls)
            counted = await run_events(api, await send(api, chat, "Count a,b,c d."))
            told = system_prompts(fake.calls[asked:])
            split = await run_events(
                api, await send(api, librarian, "Split on commas.")
            )
            again = await run_events(api, await send(api, chat, "Count a,b,c d."))
            tools = (await api.get("/api/library")).json()["tools"]
    added, granted = changes(made)
    [replaced] = changes(split)
    assert {key: added[key] for key in CHANGE_FIELDS} == {
        "what": "tool",
        "namespace": None,
        "name": "word_count",
        "before": None,
        "after": f"factory: build\nfactory_from: word_count.py\n# word_count.py\n{WORD_COUNT}",
    }
    split_row = (
        "factory: build\nfactory_from: word_count.py\nsep: ','\n# word_count.py\n"
    )
    assert (granted["what"], granted["after"]) == (
        "namespace",
        "tools:\n- word_count\n",
    )
    assert (replaced["before"], replaced["after"]) == (
        added["after"],
        split_row + SPLIT_COUNT,
    )
    assert "'done': 'added'" in kinds(made, "cell.end")[0]["output"]
    assert "'done': 'replaced'" in kinds(split, "cell.end")[0]["output"]
    outputs = [kinds(events, "cell.end")[0]["output"] for events in (counted, again)]
    assert outputs == ["2", "3"]
    assert told and all(WORD_COUNT_DOC in prompt for prompt in told)
    [row] = [tool for tool in tools if tool["name"] == "word_count"]
    assert row["granted_in"] == ["counting"]


RAISES = "def build(client, params):\n    raise RuntimeError('no index')\n"
TOOL_REFUSALS = [
    (
        library_cell("write_tool", "counter", "def build(:\n", "build"),
        "The source is not valid Python (line 1: invalid syntax).",
    ),
    (
        library_cell("write_tool", "counter", WORD_COUNT, "make"),
        "Tool counter doesn't build, so it wasn't written: ",
        "defines no 'make'",
    ),
    (
        library_cell("write_tool", "counter", RAISES, "build"),
        "build in counter.py raised RuntimeError: no index",
    ),
    (
        library_cell(
            "write_tool", "counter", "def build(client, params): return 1", "build"
        ),
        "returned int, not a Func",
    ),
    (
        library_cell("write_tool", "llm", WORD_COUNT, "build"),
        '"llm" is taken: it is a Python keyword, a built-in or app tool, or a name',
    ),
    (library_cell("write_tool", "history", WORD_COUNT, "build"), '"history" is taken'),
    (
        library_cell("write_tool", "subagent", WORD_COUNT, "build"),
        '"subagent" is taken',
    ),
    (library_cell("write_tool", "lambda", WORD_COUNT, "build"), '"lambda" is taken'),
    (
        library_cell("write_tool", "word count", WORD_COUNT, "build"),
        'Tool "word count" needs a Python name',
    ),
    (
        library_cell("write_tool", "counter", f"{WORD_COUNT}# {SERVER_KEY}\n", "build"),
        "holds a provider's key",
    ),
    (
        repl(
            "class Name(str):",
            "    pass",
            f"print(library.write_tool(Name('counter'), {WORD_COUNT!r}, 'build'))",
        ),
        "name, source and factory should be text",
    ),
    (
        repl("print(library.write_tool('counter', '#' * 1_000_001, 'build'))"),
        "may each be at most 1,000,000 characters",
    ),
]


@pytest.mark.anyio
async def test_a_refused_tool_fails_its_cell_in_plain_words_and_stores_nothing(
    library_on, connect
):
    cells = [cell for cell, *_ in TOOL_REFUSALS]
    plan = {"Try these.": [*cells, repl("FinalAnswer('refused')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            before = (await api.get("/api/library")).json()
            chat = await librarian_chat(api)
            events = await run_events(api, await send(api, chat, "Try these."))
            after = (await api.get("/api/library")).json()
    ended = kinds(events, "cell.end")[: len(TOOL_REFUSALS)]
    assert [cell["status"] for cell in ended] == ["failed"] * len(TOOL_REFUSALS)
    for cell, (_, *wordings) in zip(ended, TOOL_REFUSALS, strict=True):
        assert all(each in cell["output"] for each in wordings), cell["output"]
    assert changes(events) == []
    assert after == before
    assert events[-1]["outcome"] == "answered", events[-1]


@pytest.mark.anyio
async def test_a_tool_no_namespace_grants_never_runs_so_a_broken_one_fails_runs_only_while_granted(
    library, library_on, connect
):
    broken = dump({"factory": "build", "factory_from": "broken.py"})
    plan = {"Hi.": [repl("FinalAnswer('hello')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            with closing(open_library(library)) as connection:
                connection.execute(UPSERT_TOOL, ("broken", broken, RAISES))
            chat = await new_chat(api)
            ungranted = await run_events(api, await send(api, chat, "Hi."))
            grant = {"name": "uses_broken", "yaml": "tools: [broken]\n"}
            response = await api.post("/api/namespaces", json=grant)
            assert response.status_code == 201, response.text
            granted = await run_events(api, await send(api, chat, "Hi."))
            ungrant = {"yaml": "", "was": "tools: [broken]\n"}
            response = await api.put("/api/namespaces/uses_broken", json=ungrant)
            assert response.status_code == 200, response.text
            again = await run_events(api, await send(api, chat, "Hi."))
    outcomes = [events[-1]["outcome"] for events in (ungranted, granted, again)]
    assert outcomes == ["answered", "failed", "answered"]
    assert granted[-1]["detail"] == (
        "Tool broken doesn't build, so no chat can run while a namespace grants it:"
        " build in broken.py raised RuntimeError: no index\nIt is granted in"
        " uses_broken: take it out of their tools in the Library panel's Namespaces tab"
        " (Edit YAML)."
    )


# A tool that calls the model as write_tool's description says: with the agent's llm.
SHOUT = """from deep_reasoner.primitives import Func


def build(client, params):
    def shout(text, llm):
        \"\"\"The model's reply to text, in capitals.\"\"\"
        return llm(f"Say it back: {text}").upper()

    return Func(shout)
"""


@pytest.mark.anyio
async def test_a_tool_handed_the_agents_llm_gets_the_chats_model(
    library, library_on, connect
):
    shout = dump({"factory": "build", "factory_from": "shout.py"})
    plan = {
        "Shout hello.": [repl("print(shout('hello', llm))"), repl("FinalAnswer('ok')")],
        "Say it back: hello": ["hello there"],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            with closing(open_library(library)) as connection, writing(connection):
                connection.execute(UPSERT_TOOL, ("shout", shout, SHOUT))
            shouting = {"name": "shouting", "yaml": "tools: [shout]\n"}
            response = await api.post("/api/namespaces", json=shouting)
            assert response.status_code == 201, response.text
            chat = await chat_in(api, "shouting")
            events = await run_events(api, await send(api, chat, "Shout hello."))
    assert kinds(events, "cell.end")[0]["output"] == "HELLO THERE"
    assert {call.model for call in fake.calls} == {"gpt-6-luna"}


# The chat's messages before each of the librarian's examples is asked, each with the
# fake's cells: what its observations print comes from them.
LIBRARIANS_EARLIER = {
    "add a namespace with a worked example": {},
    "undo a change": {
        "Add a namespace for astronomy.": [
            library_cell(
                "add_namespace", "astronomy", "system_suffix: Answer in one sentence.\n"
            ),
            repl("FinalAnswer('added')"),
        ],
        "Make astronomy answer in a haiku.": [
            library_cell(
                "change_namespace", "astronomy", {"system_suffix": "Answer in a haiku."}
            ),
            repl("FinalAnswer('changed')"),
        ],
        "Let astronomy use llm.": [
            library_cell("change_namespace", "astronomy", {"tools": ["llm"]}),
            repl("FinalAnswer('granted')"),
        ],
    },
}


async def ask_as_the_librarians_example(library_on, connect, text):
    """The librarian's example *text* run through the app on the fake, after the chat's
    earlier messages: the question's events, and the namespaces after it."""
    example = parsed(text)
    question, *rest = example.messages
    earlier = LIBRARIANS_EARLIER[example.name]
    replies = [turn.content for turn in rest if turn.role == "assistant"]
    with FakeOpenAI(scripted(earlier | {question.content: replies})) as fake:
        async with connect(library_on(fake)) as api:
            chat = await librarian_chat(api)
            for prompt in earlier:
                await run_events(api, await send(api, chat, prompt))
            events = await run_events(api, await send(api, chat, question.content))
            namespaces = (await api.get("/api/library")).json()["namespaces"]
    return events, {each["name"]: each for each in namespaces}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text", LIBRARIANS_EXAMPLES, ids=lambda text: parsed(text).name
)
async def test_the_librarians_examples_show_what_the_library_tool_really_prints(
    library_on, connect, text
):
    events, _ = await ask_as_the_librarians_example(library_on, connect, text)
    shown = [turn.content for turn in parsed(text).messages[1:] if turn.role == "user"]
    printed = [
        f"<observation>\n{cell['output']}\n</observation>\n"
        for cell in kinds(events, "cell.end")[: len(shown)]
    ]
    assert printed == shown
    assert events[-1]["outcome"] == "answered", events[-1]


@pytest.mark.anyio
async def test_the_worked_example_the_librarians_example_writes_runs_as_written(
    library_on, connect
):
    [text] = [
        each
        for each in LIBRARIANS_EXAMPLES
        if parsed(each).name == "add a namespace with a worked example"
    ]
    asking, *rest = parsed(text).messages
    plan = {asking.content: [turn.content for turn in rest if turn.role == "assistant"]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            await run_events(
                api, await send(api, await librarian_chat(api), asking.content)
            )
            namespaces = (await api.get("/api/library")).json()["namespaces"]
            [written] = next(
                each for each in namespaces if each["name"] == "astronomy"
            )["decompositions"]
            question, answering = parsed(written["yaml"]).messages
            # An agent in astronomy that does just as the example it was shown does.
            plan[question.content] = [answering.content]
            chat = await chat_in(api, "astronomy")
            events = await run_events(api, await send(api, chat, question.content))
    assert written["problems"] == []
    assert events[-1]["answer"].startswith("| step | value |\n|---|---|\n| distance |")


@pytest.mark.anyio
async def test_the_librarians_undo_example_writes_the_before_back_as_it_was(
    library_on, connect
):
    [text] = [
        each for each in LIBRARIANS_EXAMPLES if parsed(each).name == "undo a change"
    ]
    events, namespaces = await ask_as_the_librarians_example(library_on, connect, text)
    [undone] = changes(events)
    assert yaml.safe_load(namespaces["astronomy"]["yaml"]) == {
        "system_suffix": "Answer in one sentence.",
        "tools": ["llm"],
    }
    assert undone["after"] == namespaces["astronomy"]["yaml"]


SPAWNED_PLAN = {
    "Have a librarian write it.": [
        repl("print(subagent('Write the example.', namespace='librarian'))"),
        repl("FinalAnswer('asked')"),
    ],
    "Hand the tool over.": [
        repl(
            "print(subagent('Write the example.', namespace='root',"
            " tools={'library': library}))"
        ),
        repl("FinalAnswer('handed')"),
    ],
    "Write the example.": [write("librarian", EXAMPLE), repl("FinalAnswer('tried')")],
}


async def spawned_run(api, chat, prompt):
    """The events of *prompt*'s run, whose root has a sub-agent write the example; the
    sub-agent's start; and its first cell, which writes."""
    events = await run_events(api, await send(api, chat, prompt))
    [child] = [each for each in kinds(events, "agent.start") if each["parent"]]
    cells = [
        each for each in kinds(events, "cell.end") if each["node"] == child["node"]
    ]
    return events, child, cells[0]


@pytest.mark.anyio
async def test_a_sub_agent_spawned_into_librarian_from_another_chat_may_not_write(
    library_on, connect
):
    with FakeOpenAI(scripted(SPAWNED_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            events, _, cell = await spawned_run(api, chat, "Have a librarian write it.")
            namespaces = (await api.get("/api/library")).json()["namespaces"]
    assert events[0]["namespace"] == "root"
    assert cell["status"] == "failed"
    assert (
        "The Library can be changed only in a chat whose namespace grants the library"
        " tool; pick librarian under the message box."
    ) in cell["output"]
    assert changes(events) == []
    assert [each["decompositions"] for each in namespaces] == FRESH_DECOMPOSITIONS
    assert events[-1]["outcome"] == "answered", events[-1]


@pytest.mark.anyio
async def test_the_librarian_may_ask_the_historian_but_never_spawn_itself(
    library_on, connect
):
    plan = {
        "Ask yourself.": [
            repl("print(subagent('Write the example.'))"),
            repl("FinalAnswer('asked')"),
        ],
    }
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await librarian_chat(api)
            events = await run_events(api, await send(api, chat, "Ask yourself."))
    [cell, _] = kinds(events, "cell.end")
    assert cell["status"] == "failed"
    assert (
        "namespace 'librarian' may not spawn a sub-agent in 'librarian'; allowed"
        " scopes: ['historian']"
    ) in cell["output"]
    assert len(kinds(events, "agent.start")) == 1


@pytest.mark.anyio
async def test_in_a_librarian_chat_a_sub_agent_handed_the_tool_elsewhere_writes(
    library_on, connect
):
    with FakeOpenAI(scripted(SPAWNED_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            # The user lets the librarian spawn anywhere, in the panel.
            listed = (await api.get("/api/library")).json()["namespaces"]
            [opened] = [each["yaml"] for each in listed if each["name"] == "librarian"]
            anywhere = {"yaml": "tools: [library]\n", "was": opened}
            response = await api.put("/api/namespaces/librarian", json=anywhere)
            assert response.status_code == 200, response.text
            chat = await librarian_chat(api)
            events, child, cell = await spawned_run(api, chat, "Hand the tool over.")
            namespaces = (await api.get("/api/library")).json()["namespaces"]
    [change] = changes(events)
    assert (child["namespace"], cell["status"]) == ("root", "done")
    assert (change["node"], change["namespace"], change["before"]) == (
        child["node"],
        "librarian",
        None,
    )
    librarian = next(each for each in namespaces if each["name"] == "librarian")
    assert librarian["decompositions"] == [
        *LIBRARIANS_SEEDED,
        {"name": "count the tickets", "yaml": EXAMPLE, "problems": EXAMPLE_PROBLEMS},
    ]


def held_on(prompt, plan):
    """A responder that follows *plan*, holding *prompt*'s first call until released;
    and the release."""
    release = threading.Event()
    follow = scripted(plan)

    def held(messages):
        if task_and_turn(messages) == (prompt, 0):
            release.wait(HELD_S)
        return follow(messages)

    return held, release


SHOWN_PLAN = {
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
    "Show me how.": [
        repl(
            "p = page.new('Why it fanned out')",
            "p.text('It split the work in **two**.')",
            "p.step(1, index=1, note='Here it fans out.')",
            "p.agent(1, history.agents(1)[0]['node'])",
            "print(p.open())",
            "p.table([{'course': 'C1', 'credits': 3}])",
            "print(p.open())",
        ),
        repl("FinalAnswer('See the page.')"),
    ],
}


@pytest.mark.anyio
async def test_a_follow_up_shows_an_earlier_messages_steps_on_a_page_its_turn_links_once(
    library_on, connect
):
    with FakeOpenAI(scripted(SHOWN_PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            namespace = {"name": "shown", "yaml": "tools: [history, page]\n"}
            assert (await api.post("/api/namespaces", json=namespace)).is_success
            chat = await chat_in(api, "shown")
            first = await run_events(
                api, await send(api, chat, "Summarize two courses.")
            )
            shown = await run_events(api, await send(api, chat, "Show me how."))
            turns = (await api.get(f"/api/chats/{chat}")).json()["turns"]
            [opened] = kinds(shown, "page.open")
            page = (await api.get(f"/api/pages/{opened['page']}")).json()
            tab = await api.get(f"/pages/{opened['page']}")
    [root] = kinds(shown, "agent.start")
    cell = kinds(shown, "cell.end")[0]
    assert {key: opened[key] for key in ("node", "cell", "title")} == {
        "node": root["node"],
        "cell": cell["cell"],
        "title": "Why it fanned out",
    }
    assert cell["output"] == (
        'Opened the page "Why it fanned out"; the chat links it under your answer.\n'
        'Saved the page "Why it fanned out" again, as it stands now; the chat still'
        " links it."
    )
    assert kinds(turns[1]["events"], "page.open") == [opened]
    assert shown[-1]["outcome"] == "answered", shown[-1]
    text, step, agent, table = page["blocks"]
    top, *sub_agents = kinds(first, "agent.start")
    fanned = next(
        each for each in kinds(first, "cell.end") if each["node"] == top["node"]
    )
    assert page["title"] == "Why it fanned out"
    assert text == {"type": "text", "text": "It split the work in **two**."}
    assert (step["note"], step["step"]["code"], step["step"]["output"]) == (
        "Here it fans out.",
        fanned["code"],
        fanned["output"],
    )
    assert [row["task"] for row in step["step"]["agents"]] == [
        "Summarize C1.",
        "Summarize C2.",
    ]
    assert [row["node"] for row in agent["agents"]] == [sub_agents[0]["node"]]
    assert [each["code"] for each in agent["agents"][0]["steps"]] == [
        "print('C1 has 3 credits')",
        "FinalAnswer('C1: 3')",
    ]
    assert (table["columns"], table["rows"]) == (["course", "credits"], [["C1", "3"]])
    assert tab.status_code == 200
    assert PAGE_DESCRIPTION in fake.calls[-1].messages[0]["content"]


@pytest.mark.anyio
async def test_a_second_message_while_a_run_is_going_is_refused(library_on, connect):
    held, release = held_on("Count to two.", PLAN)
    with FakeOpenAI(held) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            run = await send(api, chat, "Count to two.")
            refused = await api.post(
                f"/api/chats/{chat}/runs", json={"prompt": "What did I ask?"}
            )
            stopped = await api.post(f"/api/runs/{run}/stop")
            events = await run_events(api, run)
            release.set()
            again = await api.post(f"/api/runs/{run}/stop")
    assert refused.status_code == 409
    assert refused.json() == {"detail": f"Chat {chat} has a run going."}
    assert stopped.status_code == 200
    assert events[-1]["outcome"] == "stopped"
    assert again.status_code == 409


@pytest.mark.parametrize("chat", ["a chat!", "x" * 101])
def test_a_bad_chat_id_is_refused(client, chat):
    assert client.get(f"/api/chats/{chat}").status_code == 422
    assert client.patch(f"/api/chats/{chat}", json={"title": "Hi"}).status_code == 422
    assert client.delete(f"/api/chats/{chat}").status_code == 422
    response = client.post(f"/api/chats/{chat}/runs", json={"prompt": "Hi."})
    assert response.status_code == 422


def test_an_empty_message_is_refused(client):
    chat = client.post("/api/chats").json()["id"]
    response = client.post(f"/api/chats/{chat}/runs", json={"prompt": ""})
    assert response.status_code == 422


def test_a_chat_that_is_not_there_is_refused(client):
    chat = "no-such-chat"
    assert client.get(f"/api/chats/{chat}").status_code == 404
    assert client.patch(f"/api/chats/{chat}", json={"title": "Hi"}).status_code == 404
    assert client.delete(f"/api/chats/{chat}").status_code == 404
    response = client.post(f"/api/chats/{chat}/runs", json={"prompt": "Hi."})
    assert response.status_code == 404
    assert response.json() == {"detail": "There is no chat no-such-chat."}


@pytest.mark.parametrize(
    ("run", "status"), [("20261025-120000-000000", 404), ("not-a-run", 422)]
)
def test_a_run_that_is_not_there_is_refused(client, run, status):
    assert client.get(f"/api/runs/{run}/events").status_code == status
    assert client.post(f"/api/runs/{run}/stop").status_code == status


def test_new_chats_are_listed_newest_first(client):
    first, second = client.post("/api/chats"), client.post("/api/chats")
    assert first.status_code == second.status_code == 201
    assert first.json()["title"] == "New chat" and first.json()["live"] is None
    assert first.json()["id"] != second.json()["id"]
    assert client.get("/api/chats").json() == [second.json(), first.json()]


def test_a_chat_is_renamed_with_its_title_trimmed(client):
    chat = client.post("/api/chats").json()
    response = client.patch(
        f"/api/chats/{chat['id']}", json={"title": "  Trip to Rome \n"}
    )
    assert response.status_code == 200
    assert response.json() == chat | {"title": "Trip to Rome"}
    assert client.get("/api/chats").json() == [response.json()]


@pytest.mark.parametrize("title", ["", " \n "])
def test_a_blank_title_is_refused(client, title):
    chat = client.post("/api/chats").json()
    response = client.patch(f"/api/chats/{chat['id']}", json={"title": title})
    assert response.status_code == 422
    assert client.get("/api/chats").json() == [chat]


def test_a_title_past_the_longest_kept_is_cut(client):
    chat = client.post("/api/chats").json()
    long = "Trip " + "x" * TITLE_CHARS
    renamed = client.patch(f"/api/chats/{chat['id']}", json={"title": long})
    assert renamed.json()["title"] == long[:TITLE_CHARS]


async def title_of(api, chat):
    return (await api.get(f"/api/chats/{chat}")).json()["title"]


@pytest.mark.anyio
async def test_the_first_message_titles_a_chat_cut_past_the_longest_kept(
    library_on, connect
):
    prompt = "Count to two. " + "x" * TITLE_CHARS + "\nSlowly."
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            first = await api.post(f"/api/chats/{chat}/runs", json={"prompt": prompt})
            await run_events(api, first.json()["run"])
            later = await api.post(f"/api/chats/{chat}/runs", json={"prompt": "Hi."})
            await run_events(api, later.json()["run"])
            title = await title_of(api, chat)
    assert title == first.json()["title"] == later.json()["title"]
    assert title == prompt[:TITLE_CHARS]


@pytest.mark.anyio
async def test_chats_are_listed_by_their_latest_message_else_when_made(
    library_on, connect
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            older, newer = await new_chat(api), await new_chat(api)
            run = await send(api, older, "Count to two.")
            await run_events(api, run)
            used = (await api.get("/api/chats")).json()
            newest = await new_chat(api)
            listed = (await api.get("/api/chats")).json()
            read = (await api.get(f"/api/chats/{older}")).json()
    assert [each["id"] for each in used] == [older, newer]
    assert [(each["id"], each["last_run"]) for each in listed] == [
        (newest, None),
        (older, run),
        (newer, None),
    ]
    assert read["last_run"] == run


@pytest.mark.anyio
async def test_the_first_message_titles_a_new_chat_and_a_later_one_does_not(
    library_on, connect
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            await run_events(api, await send(api, chat, "Count to two.\nSlowly."))
            titled = await title_of(api, chat)
            await run_events(api, await send(api, chat, "What did I ask?"))
            later = await title_of(api, chat)
    assert titled == later == "Count to two."


@pytest.mark.anyio
async def test_a_renamed_chat_keeps_its_name_when_its_first_message_runs(
    library_on, connect
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            await api.patch(f"/api/chats/{chat}", json={"title": "Counting"})
            await run_events(api, await send(api, chat, "Count to two."))
            title = await title_of(api, chat)
    assert title == "Counting"


@pytest.mark.anyio
async def test_a_chat_reads_back_with_its_turns_in_order(library_on, connect):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            chat, another = await new_chat(api), await new_chat(api)
            first = await send(api, chat, "Count to two.")
            first_events = await run_events(api, first)
            await run_events(api, await send(api, another, "Count to two."))
            second = await send(api, chat, "What did I ask?")
            second_events = await run_events(api, second)
            response = await api.get(f"/api/chats/{chat}")
    read = response.json()
    assert (read["id"], read["title"], read["live"]) == (chat, "Count to two.", None)
    assert read["turns"] == [
        {"run": first, "events": first_events},
        {"run": second, "events": second_events},
    ]


@pytest.mark.anyio
async def test_a_reload_mid_run_reads_the_turn_so_far_and_streams_the_rest(
    library_on, connect
):
    held, release = held_on("Count to two.", PLAN)
    with FakeOpenAI(held) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            run = await send(api, chat, "Count to two.")
            read = (await api.get(f"/api/chats/{chat}")).json()
            so_far = read["turns"][0]["events"]
            release.set()
            url = f"/api/runs/{run}/events"
            rest = await api.get(url, params={"after": so_far[-1]["seq"]})
            whole = await run_events(api, run)
    assert read["live"] == run and whole[-1]["kind"] == "turn.end"
    assert so_far + [event for _, event in server_sent(rest.text)] == whole


@pytest.mark.anyio
async def test_a_run_that_is_going_shows_in_the_list_and_in_its_chat(
    library_on, connect
):
    held, release = held_on("Count to two.", PLAN)
    with FakeOpenAI(held) as fake:
        async with connect(library_on(fake)) as api:
            chat, idle = await new_chat(api), await new_chat(api)
            run = await send(api, chat, "Count to two.")
            listed = (await api.get("/api/chats")).json()
            read = (await api.get(f"/api/chats/{chat}")).json()
            release.set()
            await run_events(api, run)
            after = (await api.get("/api/chats")).json()
    assert {each["id"]: each["live"] for each in listed} == {chat: run, idle: None}
    assert read["live"] == run and [turn["run"] for turn in read["turns"]] == [run]
    assert read["turns"][0]["events"][0]["kind"] == "turn.start"
    assert {each["id"]: each["live"] for each in after} == {chat: None, idle: None}


@pytest.mark.anyio
async def test_deleting_a_chat_stops_its_run_and_removes_its_runs_events_and_folders(
    library_on, home, data
):
    held, release = held_on("What did I ask?", PLAN)
    with FakeOpenAI(held) as fake:
        library = library_on(fake)
        app = create_app(library, home)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=LOCAL) as api:
            chat, kept = await new_chat(api), await new_chat(api)
            ended = await send(api, chat, "Count to two.")
            await run_events(api, ended)
            other = await send(api, kept, "Count to two.")
            await run_events(api, other)
            going = await send(api, chat, "What did I ask?")
            deleted = await api.delete(f"/api/chats/{chat}")
            release.set()
            listed = (await api.get("/api/chats")).json()
            read = await api.get(f"/api/chats/{chat}")
            kept_read = (await api.get(f"/api/chats/{kept}")).json()
    assert deleted.status_code == 204
    assert app.state.runs.live(chat) is None
    assert [each["id"] for each in listed] == [kept]
    assert read.status_code == 404
    assert saved_runs(library) == {other}
    assert [turn["run"] for turn in kept_read["turns"]] == [other]
    runs = data / "runs"
    assert not (runs / ended).exists() and not (runs / going).exists()
    assert (runs / other).is_dir()


@pytest.mark.anyio
async def test_a_deleted_chats_messages_leave_nothing_in_the_app_file(
    library_on, connect
):
    secret = "Remember the passphrase heliotrope-quokka."
    with FakeOpenAI(scripted({secret: [repl("FinalAnswer('remembered')")]})) as fake:
        library = library_on(fake)
        async with connect(library) as api:
            chat = await new_chat(api)
            await run_events(api, await send(api, chat, secret))
            held = b"heliotrope-quokka" in library.read_bytes()
            deleted = await api.delete(f"/api/chats/{chat}")
    assert held and deleted.status_code == 204
    assert b"heliotrope-quokka" not in library.read_bytes()


@pytest.mark.anyio
async def test_chats_survive_a_restart(library_on, connect):
    with FakeOpenAI(scripted(PLAN)) as fake:
        library = library_on(fake)
        async with connect(library) as api:
            chat = await new_chat(api)
            run = await send(api, chat, "Count to two.")
            events = await run_events(api, run)
            renamed = await api.patch(f"/api/chats/{chat}", json={"title": "Counting"})
        async with connect(library) as restarted:
            listed = (await restarted.get("/api/chats")).json()
            read = (await restarted.get(f"/api/chats/{chat}")).json()
    assert listed == [renamed.json()]
    assert read == renamed.json() | {"turns": [{"run": run, "events": events}]}


def test_a_copy_of_the_app_file_under_another_home_has_its_chats_and_settings(
    client, home, library, tmp_path
):
    (home / "code").mkdir()
    client.post("/api/servers", json={"name": "vllm", "base_url": VLLM, "key": "sk-1"})
    client.put("/api/default-folder", json={"folder": "~/code"})
    chat, other = client.post("/api/chats").json(), client.post("/api/chats").json()
    qwen = {"provider": "vllm", "name": "qwen3"}
    picks = {"namespace": "course_advisor", "model": qwen}
    client.patch(f"/api/chats/{chat['id']}", json=picks)
    client.patch(f"/api/chats/{other['id']}", json={"title": "Plans"})
    here = client.get("/api/chats").json()
    copy = tmp_path / "elsewhere.sqlite"
    shutil.copy(library, copy)
    with TestClient(create_app(copy, tmp_path / "elsewhere"), base_url=LOCAL) as moved:
        there = moved.get("/api/chats").json()
        server = by_name(moved.get("/api/providers").json())["vllm"]
        folder = moved.get("/api/default-folder").json()

    def kept(chats):
        fields = ("id", "title", "namespace", "model", "folder")
        return [tuple(each[name] for name in fields) for each in chats]

    assert (
        kept(there)
        == kept(here)
        == [
            (other["id"], "Plans", "router", qwen, "~/code"),
            (chat["id"], "New chat", "course_advisor", qwen, "~/code"),
        ]
    )
    assert (server["base_url"], server["key"]) == (VLLM, "none")
    assert folder == {"folder": "~/code", "missing": False}


@pytest.mark.anyio
async def test_a_copy_of_the_app_file_under_another_home_shows_the_chats_past_turns(
    library_on, connect, tmp_path
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        library = library_on(fake)
        async with connect(library) as api:
            chat = await new_chat(api)
            runs = [await send(api, chat, "Count to two.")]
            await run_events(api, runs[0])
            runs.append(await send(api, chat, "What did I ask?"))
            await run_events(api, runs[1])
            here = (await api.get(f"/api/chats/{chat}")).json()
    # With no run going, the app keeps nothing of the file's beside it.
    assert [each.name for each in library.parent.glob(f"{library.name}-*")] == []
    copy = tmp_path / "elsewhere.sqlite"
    shutil.copy(library, copy)
    elsewhere = httpx.ASGITransport(app=create_app(copy, tmp_path / "elsewhere"))
    async with httpx.AsyncClient(transport=elsewhere, base_url=LOCAL) as api:
        there = (await api.get(f"/api/chats/{chat}")).json()
    assert [turn["run"] for turn in there["turns"]] == runs
    assert there["turns"] == here["turns"]
    assert there["turns"][1]["events"][-1]["answer"] == "to count"


def test_a_run_the_app_stopped_under_ends_failed_when_the_app_next_opens_the_file(
    imported, home
):
    chat = create_chat(imported)["id"]
    run = "20261025-120000-000000"
    started = {"kind": "turn.start", "run": run, "chat": chat, "prompt": "Hi."}
    with closing(open_library(imported)) as connection:
        save_event(connection, run, {"seq": 1, "t": 100.0} | started)
    with TestClient(create_app(imported, home), base_url=LOCAL) as client:
        turns = client.get(f"/api/chats/{chat}").json()["turns"]
    events = turns[0]["events"]
    assert [event["kind"] for event in events] == ["turn.start", "turn.end"]
    assert (events[-1]["outcome"], events[-1]["detail"]) == ("failed", LOST)


GOOD = "20261025-120000-000000"
BROKEN = "20261025-120001-000000"
STARTED = row(1, "turn.start", prompt="Hi.")
USED = row(2, "usage", node=1, cost=0.25)


@pytest.mark.parametrize(
    "rows, ends",
    [
        pytest.param([(1, "{not json")], False, id="invalid JSON"),
        pytest.param([], False, id="no events"),
        pytest.param(
            [STARTED, row(3, "usage", cost=0.5)], True, id="a gap in its seqs"
        ),
        pytest.param([STARTED, (2, "[1, 2]")], True, id="a last event no object"),
        pytest.param(
            [(1, json.dumps({"seq": 1, "kind": "turn.start", "prompt": "Hi."})), USED],
            True,
            id="a first event with no time",
        ),
        pytest.param(
            [STARTED, (2, '{"seq": 2, "t": 1e999, "kind": "usage", "cost": 0.5}')],
            True,
            id="a time past range",
        ),
        pytest.param(
            [
                STARTED,
                (2, '{"seq": 2, "t": 102, "kind": "usage", "cost": 9%s}' % ("9" * 400)),
            ],
            True,
            id="a cost of 400 digits",
        ),
        pytest.param(
            [STARTED, row(2**63 - 1, "usage", cost=0.5)],
            False,
            id="a seq at SQLite's largest",
        ),
        pytest.param(
            [STARTED, (1.5, '{"seq": 1.5, "t": 101.5, "kind": "usage"}')],
            True,
            id="a seq that isn't whole",
        ),
    ],
)
def test_the_app_starts_past_a_run_it_cannot_end_and_ends_the_rest(
    imported, home, rows, ends
):
    chat = create_chat(imported)["id"]
    plant(imported, chat, GOOD, [STARTED, USED])
    plant(imported, chat, BROKEN, rows)
    for _ in range(2):
        with TestClient(create_app(imported, home), base_url=LOCAL) as client:
            turns = client.get(f"/api/chats/{chat}").json()["turns"]
    good = turns[0]["events"]
    assert (turns[0]["run"], good[-1]["outcome"], good[-1]["detail"]) == (
        GOOD,
        "failed",
        LOST,
    )
    assert (good[-1]["seq"], good[-1]["cost"], good[-1]["seconds"]) == (3, 0.25, 1)
    assert (ends_saved(imported, GOOD), ends_saved(imported, BROKEN)) == (1, ends)


def test_the_app_starts_on_a_file_it_cannot_write_and_shows_a_lost_run_failed(
    imported, home, monkeypatch
):
    chat = create_chat(imported)["id"]
    plant(imported, chat, GOOD, [STARTED, USED])
    connect = sqlite3.connect

    def read_only(path, **options):
        return connect(f"{Path(path).as_uri()}?mode=ro", uri=True, **options)

    monkeypatch.setattr(sqlite3, "connect", read_only)
    with TestClient(create_app(imported, home), base_url=LOCAL) as client:
        turns = client.get(f"/api/chats/{chat}").json()["turns"]
    assert turns[0]["events"][-1]["detail"] == LOST
    monkeypatch.undo()
    assert ends_saved(imported, GOOD) == 0


def test_a_lost_run_is_ended_once_however_often_the_app_starts(imported, home):
    chat = create_chat(imported)["id"]
    plant(imported, chat, GOOD, [STARTED, (5, '{"seq": 5, "kind": "usage"}')])
    for _ in range(2):
        with TestClient(create_app(imported, home), base_url=LOCAL):
            pass
    with closing(open_library(imported)) as connection:
        seqs = [seq for (seq,) in connection.execute("SELECT seq FROM events")]
    assert (ends_saved(imported, GOOD), seqs) == (1, [1, 5, 6])


def test_a_run_with_no_turn_end_reads_back_failed_without_one_being_saved(client):
    chat = client.post("/api/chats").json()["id"]
    plant(client.app.state.runs.library, chat, GOOD, [STARTED, USED])
    events = client.get(f"/api/chats/{chat}").json()["turns"][0]["events"]
    streamed = client.get(f"/api/runs/{GOOD}/events", params={"after": 1})
    assert [event["kind"] for event in events] == ["turn.start", "usage", "turn.end"]
    assert (events[-1]["seq"], events[-1]["outcome"]) == (3, "failed")
    assert [event for _, event in server_sent(streamed.text)] == events[1:]
    assert ends_saved(client.app.state.runs.library, GOOD) == 0


def test_an_event_row_the_app_could_not_have_written_is_left_out(client):
    chat = client.post("/api/chats").json()["id"]
    not_a_number = '{"seq": 3, "t": 103, "kind": "usage", "cost": NaN}'
    rows = [
        STARTED,
        (2, "{oops"),
        (3, not_a_number),
        row(4, "usage"),
        row(5, "turn.end"),
    ]
    plant(client.app.state.runs.library, chat, GOOD, rows)
    read = client.get(f"/api/chats/{chat}")
    streamed = client.get(f"/api/runs/{GOOD}/events")
    assert read.status_code == streamed.status_code == 200
    events = read.json()["turns"][0]["events"]
    assert [event["seq"] for event in events] == [1, 4, 5]
    assert [event for _, event in server_sent(streamed.text)] == events


def test_an_event_numbered_below_one_is_left_out_so_its_run_still_reads(client):
    chat = client.post("/api/chats").json()["id"]
    before_its_start = [row(-5, "think", node=0, text="first"), row(0, "think", node=0)]
    plant(
        client.app.state.runs.library,
        chat,
        GOOD,
        [*before_its_start, STARTED, row(2, "turn.end", outcome="answered")],
    )
    turns = client.get(f"/api/chats/{chat}").json()["turns"]
    assert [
        (turn["run"], [event["seq"] for event in turn["events"]]) for turn in turns
    ] == [(GOOD, [1, 2])]


def answered(seq, answer):
    return row(seq, "turn.end", outcome="answered", answer=answer, detail=None)


@pytest.mark.anyio
async def test_the_page_and_a_follow_ups_turns_agree_on_how_each_run_started_and_ended(
    library_on, connect
):
    junk_after_its_end = "20261025-120005-000000"
    with FakeOpenAI(scripted(PLAN)) as fake:
        library = library_on(fake)
        async with connect(library) as api:
            chat = await new_chat(api)
            plant(
                library,
                chat,
                GOOD,
                [row(1, "turn.start", prompt="Count to two."), answered(2, "2")],
            )
            plant(library, chat, "20261025-120001-000000", [])
            plant(library, chat, "20261025-120002-000000", [USED])
            plant(library, chat, "20261025-120003-000000", [row(1, "turn.start")])
            plant(
                library,
                chat,
                "20261025-120004-000000",
                [row(1, "turn.start", prompt=5)],
            )
            ended_then_junk = [
                row(1, "turn.start", prompt="Again."),
                answered(2, "3"),
                (3, "{oops"),
            ]
            plant(library, chat, junk_after_its_end, ended_then_junk)
            page = (await api.get(f"/api/chats/{chat}")).json()["turns"]
            sent = await api.post(
                f"/api/chats/{chat}/runs", json={"prompt": "What did I ask?"}
            )
            events = await run_events(api, sent.json()["run"])
    assert sent.status_code == 201 and events[-1]["outcome"] == "answered"
    assert [turn["run"] for turn in page] == [GOOD, junk_after_its_end]
    assert [turn["events"][-1]["answer"] for turn in page] == ["2", "3"]
    told = fake.calls[-1].messages[-6:]
    assert [(message["role"], message["content"]) for message in told] == [
        ("system", EARLIER_TURNS_NOTE),
        ("user", "Count to two."),
        ("assistant", folded_answer("'2'")),
        ("user", "Again."),
        ("assistant", folded_answer("'3'")),
        ("user", "What did I ask?"),
    ]


TEXT_ONLY = [{"type": "text", "text": "It **counted**."}]


def a_page(library, chat):
    """A page of *chat*'s, as the page tool saves one; its id."""
    with closing(open_library(library)) as connection:
        return add_page(connection, chat, GOOD, 1, "Counted", json.dumps(TEXT_ONLY))


def pages_in(library):
    with closing(open_library(library)) as connection:
        return {page for (page,) in connection.execute("SELECT id FROM pages")}


def test_the_chat_and_a_pages_tab_load_nothing_but_from_the_app(client):
    page = a_page(client.app.state.runs.library, client.post("/api/chats").json()["id"])
    for path in ["/", f"/pages/{page}"]:
        policy = client.get(path).headers["content-security-policy"]
        assert "default-src 'self'" in policy
        assert "img-src 'self' data:" in policy


def test_a_page_is_read_from_the_api_and_its_tab_is_the_page_drawing_it(client):
    library = client.app.state.runs.library
    page = a_page(library, client.post("/api/chats").json()["id"])
    read = client.get(f"/api/pages/{page}")
    tab = client.get(f"/pages/{page}")
    assert read.json() == {"id": page, "title": "Counted", "blocks": TEXT_ONLY}
    assert tab.headers["content-type"].startswith("text/html")
    assert tab.text == (WEB / "page.html").read_text()


@pytest.mark.parametrize("page", ["no-such-page", "not%20an%20id", "x" * 101])
def test_a_page_not_there_or_with_no_pages_id_is_a_404_its_tab_says_in_the_apps_look(
    client, page
):
    read, tab = client.get(f"/api/pages/{page}"), client.get(f"/pages/{page}")
    assert (read.status_code, tab.status_code) == (404, 404)
    assert read.json()["detail"] == NO_PAGE
    assert tab.text == (WEB / "page.html").read_text()
    assert "default-src 'self'" in tab.headers["content-security-policy"]


@pytest.mark.parametrize("path", ["/api/pages/{}", "/pages/{}"])
def test_a_page_named_from_another_host_or_site_is_refused(client, path):
    library = client.app.state.runs.library
    url = path.format(a_page(library, client.post("/api/chats").json()["id"]))
    rebound = client.get(url, headers={"Host": "rebound.example"})
    cross = client.get(url, headers={"Origin": "https://elsewhere.example"})
    assert (rebound.status_code, cross.status_code) == (400, 403)


def test_a_page_whose_record_nests_deep_is_served(client):
    library = client.app.state.runs.library
    drive = {"task": "Go.", "status": "done", "unpriced": False, "steps": []}
    for _ in range(1_000):
        cell = {"kind": "cell", "code": "", "status": "done", "agents": [drive]}
        drive = drive | {"steps": [cell]}
    blocks = [{"type": "agent", "turn": 1, "node": 2, "agents": [drive]}]
    with closing(open_library(library)) as connection:
        page = add_page(connection, "a-chat", GOOD, 1, "Deep", json.dumps(blocks))
    assert client.get(f"/api/pages/{page}").status_code == 200


def plant_page(library, blocks_text):
    """A page row holding *blocks_text* as it is, as a file from someone else might; its
    id."""
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute("DELETE FROM pages WHERE id = 'planted'")
        connection.execute(
            "INSERT INTO pages VALUES ('planted', 'a-chat', ?, 1, 'Planted', ?, 0)",
            (GOOD, blocks_text),
        )
    return "planted"


def deep_agent_text(levels):
    """An agent block whose sub-agent nests *levels* deep, as JSON written by hand, as
    json.dumps may not reach that deep."""
    drive = '{"task": "Go.", "status": "done", "unpriced": false, "steps": '
    cell = '[{"kind": "cell", "code": "", "status": "done", "agents": ['
    chain = (drive + cell) * levels + drive + "[]}" + "]}]}" * levels
    return '[{"type": "agent", "turn": 1, "node": 2, "agents": [' + chain + "]}]"


def test_a_page_nested_about_as_deep_as_json_reads_is_served_or_a_404_never_a_500(
    client,
):
    library = client.app.state.runs.library
    # Around 10,000 deep, JSON reads a little deeper than it writes back.
    statuses = {
        client.get(
            f"/api/pages/{plant_page(library, deep_agent_text(levels))}"
        ).status_code
        for levels in range(2480, 2520)
    }
    assert statuses == {200, 404}


def test_a_pages_tab_opens_on_any_page_there_and_the_api_says_whether_it_can_be_shown(
    client,
):
    page = plant_page(client.app.state.runs.library, "not json")
    assert client.get(f"/pages/{page}").status_code == 200
    assert client.get(f"/api/pages/{page}").status_code == 404


def test_deleting_a_chat_deletes_its_pages_and_no_other(client):
    library = client.app.state.runs.library
    chat, other = (client.post("/api/chats").json()["id"] for _ in range(2))
    a_page(library, chat)
    kept = a_page(library, other)
    deleted = client.delete(f"/api/chats/{chat}")
    assert deleted.status_code == 204
    assert pages_in(library) == {kept}


def test_pages_of_chats_deleted_before_their_pages_are_deleted_when_the_app_starts(
    imported, home
):
    kept = a_page(imported, create_chat(imported)["id"])
    a_page(imported, "a-deleted-chat")
    with TestClient(create_app(imported, home), base_url=LOCAL):
        pass
    assert pages_in(imported) == {kept}


def test_runs_of_chats_deleted_before_their_runs_are_deleted_when_the_app_starts(
    imported, home
):
    chat = create_chat(imported)["id"]
    plant(imported, chat, GOOD, [STARTED, row(2, "turn.end")])
    plant(imported, "a-deleted-chat", BROKEN, [STARTED, row(2, "turn.end")])
    with closing(open_library(imported)) as connection:
        connection.execute(
            "INSERT INTO chats (id, title, created) VALUES (NULL, '', 0)"
        )
    with TestClient(create_app(imported, home), base_url=LOCAL):
        pass
    assert saved_runs(imported) == {GOOD}


def test_a_shared_files_own_trigger_never_keeps_what_deleting_a_chat_removes(client):
    library = client.app.state.runs.library
    chat = client.post("/api/chats").json()["id"]
    plant(library, chat, GOOD, [STARTED, row(2, "turn.end")])
    keeping = """
        CREATE TABLE keep (data TEXT);
        CREATE TRIGGER keeping BEFORE DELETE ON events
        BEGIN INSERT INTO keep VALUES (old.data); END;
    """
    with closing(sqlite3.connect(library)) as connection:
        connection.executescript(keeping)
    deleted = client.delete(f"/api/chats/{chat}")
    with closing(sqlite3.connect(library)) as connection:
        kept = connection.execute("SELECT count(*) FROM keep").fetchone()[0]
    assert deleted.status_code == 204
    assert (kept, saved_runs(library)) == (0, set())


@pytest.mark.anyio
async def test_a_providers_key_from_the_shell_that_a_cell_prints_never_reaches_the_file(
    library_on, connect, monkeypatch
):
    shell_key = "sk-or-from-the-shell-0123456789"
    monkeypatch.setenv("OPENROUTER_API_KEY", shell_key)
    names = (
        f"os.environ['OPENROUTER_API_KEY'], os.environ['{server_key_env(SERVER_NAME)}']"
    )
    plan = {
        "Show the keys.": [
            repl("import os", f"print({names})"),
            repl("FinalAnswer('shown')"),
        ]
    }
    with FakeOpenAI(scripted(plan)) as fake:
        library = library_on(fake)
        async with connect(library) as api:
            events = await run_events(
                api, await send(api, await new_chat(api), "Show the keys.")
            )
    assert cell_outputs(events)[0] == "[key withheld] [key withheld]"
    held = library.read_bytes()
    assert shell_key.encode() not in held and SERVER_KEY.encode() not in held


@pytest.mark.anyio
async def test_deleting_a_chat_in_a_copy_of_the_file_leaves_the_originals_runs_and_scratch(
    library_on, connect, data, tmp_path
):
    with FakeOpenAI(scripted(WRITE_THEN_READ)) as fake:
        library = library_on(fake)
        async with connect(library) as api:
            chat = await new_chat(api)
            run = await send(api, chat, "Write a note.")
            await run_events(api, run)
        copy = tmp_path / "copy.sqlite"
        shutil.copy(library, copy)
        async with connect(copy) as api:
            copied = (await api.get(f"/api/chats/{chat}")).json()
            deleted = await api.delete(f"/api/chats/{chat}")
        async with connect(library) as api:
            original = (await api.get(f"/api/chats/{chat}")).json()
    assert [turn["run"] for turn in copied["turns"]] == [run]
    assert deleted.status_code == 204
    assert [turn["run"] for turn in original["turns"]] == [run]
    assert (data / "runs" / run / "spec.yaml").is_file()
    assert (data / "work" / chat / "note.txt").read_text() == "from message 1"


def test_a_library_file_that_cannot_be_written_is_refused_in_plain_words(
    client, monkeypatch
):
    chat = client.post("/api/chats").json()
    connect = sqlite3.connect

    def read_only(path, **options):
        return connect(f"{Path(path).as_uri()}?mode=ro", uri=True, **options)

    monkeypatch.setattr(sqlite3, "connect", read_only)
    added = client.post("/api/chats")
    folder = client.put("/api/default-folder", json={"folder": None})
    assert (added.status_code, folder.status_code) == (400, 400)
    assert added.json() == {
        "detail": "The Library file can't be written: attempt to write a readonly database."
    }
    assert client.get("/api/chats").json() == [chat]


def test_a_broken_library_file_is_not_passed_off_as_one_that_cannot_be_written(
    client, library
):
    client.post("/api/chats")
    with closing(open_library(library)) as connection:
        connection.execute("DROP TABLE chats")
    with pytest.raises(sqlite3.OperationalError, match="no such table: chats"):
        client.get("/api/chats")


def by_name(rows, field="name"):
    return {each[field]: each for each in rows}


def closed_port():
    with socket.create_server(("127.0.0.1", 0)) as probe:
        return probe.getsockname()[1]


def test_a_saved_key_comes_back_as_a_status_never_as_itself(client, home):
    saved = client.put("/api/providers/OpenAI/key", json={"key": SAVED_KEY})
    listed = client.get("/api/providers")
    assert saved.status_code == listed.status_code == 200
    assert by_name(listed.json())["OpenAI"] == {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "custom": False,
        "key": "set",
    }
    assert SAVED_KEY not in saved.text and SAVED_KEY not in listed.text
    env = home / ENV_FILE
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_text() == f"OPENAI_API_KEY='{SAVED_KEY}'\n"


def test_keys_saved_through_the_app_never_land_in_the_app_file(client, library):
    client.put("/api/providers/OpenAI/key", json={"key": SAVED_KEY})
    server = {"name": "vllm", "base_url": VLLM, "key": "sk-vllm-first"}
    client.post("/api/servers", json=server)
    client.put("/api/servers/vllm", json={"base_url": VLLM, "key": "sk-vllm-second"})
    listed = by_name(client.get("/api/providers").json())
    assert listed["OpenAI"]["key"] == listed["vllm"]["key"] == "set"
    held = library.read_bytes()
    for key in (SAVED_KEY, "sk-vllm-first", "sk-vllm-second"):
        assert key.encode() not in held


def test_a_server_planted_under_a_vendors_name_never_gets_its_key(
    client, library, monkeypatch
):
    monkeypatch.setenv("DEEP_REASONING_ANTHROPIC_KEY", SHELL_KEY)
    with FakeOpenAI(scripted(PLAN)) as anthropic, FakeOpenAI(scripted(PLAN)) as planted:
        # Anthropic's own address, faked, so its listing stays on this computer.
        vendors = [
            replace(each, base_url=anthropic.base_url)
            if each.name == "Anthropic"
            else each
            for each in VENDORS
        ]
        monkeypatch.setattr(settings_module, "VENDORS", tuple(vendors))
        entry = {"name": "anthropic", "base_url": planted.base_url}
        with closing(open_library(library)) as connection:
            connection.execute(
                "INSERT INTO app_settings VALUES (?, ?)", (SERVERS, json.dumps([entry]))
            )
        providers = client.get("/api/providers").json()
        listings = client.get("/api/models").json()
    assert planted.listed == [] and anthropic.listed == [SHELL_KEY]
    assert [each["name"] for each in providers if each["custom"]] == []
    assert [each["provider"] for each in listings] == ["Anthropic"]


def shared_copy(library, tmp_path, name, base_url):
    """A copy of the file, as someone else might share it, whose only server is *name*
    at *base_url*."""
    shared = tmp_path / "shared.sqlite"
    shutil.copy(library, shared)
    theirs = [{"name": name, "base_url": base_url}]
    with closing(open_library(shared)) as connection:
        connection.execute(
            "UPDATE app_settings SET json = ? WHERE key = ?",
            (json.dumps(theirs), SERVERS),
        )
    return shared


def test_a_shared_files_server_of_the_same_name_never_gets_the_users_key(
    client, home, library, tmp_path
):
    with FakeOpenAI(scripted(PLAN)) as own, FakeOpenAI(scripted(PLAN)) as planted:
        mine = {"name": "groq", "base_url": own.base_url, "key": SAVED_KEY}
        client.post("/api/servers", json=mine)
        shared = shared_copy(library, tmp_path, "GROQ", planted.base_url)
        with TestClient(create_app(shared, home), base_url=LOCAL) as other:
            status = by_name(other.get("/api/providers").json())["GROQ"]["key"]
            blank = {"base_url": planted.base_url}
            saved = by_name(other.put("/api/servers/GROQ", json=blank).json())
            other.get("/api/models")
            other.delete("/api/providers/GROQ/key")
            other.delete("/api/servers/GROQ")
        still = by_name(client.get("/api/providers").json())["groq"]["key"]
        client.get("/api/models")
    assert planted.listed == [None]
    assert status == saved["GROQ"]["key"] == "none"
    assert own.listed == [SAVED_KEY] and still == "set"
    assert saved_keys(home)["DEEP_REASONING_GROQ_KEY"] == SAVED_KEY


def test_a_key_typed_for_a_shared_files_server_of_the_same_name_goes_to_it(
    client, home, library, tmp_path
):
    with FakeOpenAI(scripted(PLAN)) as own, FakeOpenAI(scripted(PLAN)) as planted:
        mine = {"name": "groq", "base_url": own.base_url, "key": SAVED_KEY}
        client.post("/api/servers", json=mine)
        shared = shared_copy(library, tmp_path, "GROQ", planted.base_url)
        with TestClient(create_app(shared, home), base_url=LOCAL) as other:
            typed = {"base_url": planted.base_url, "key": "sk-typed-for-theirs"}
            saved = by_name(other.put("/api/servers/GROQ", json=typed).json())
            other.get("/api/models")
    assert saved["GROQ"]["key"] == "set"
    assert planted.listed == ["sk-typed-for-theirs"]


def test_removing_a_saved_key_leaves_the_shells_in_use(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SHELL_KEY)
    shell = client.get("/api/providers")
    client.put("/api/providers/OpenAI/key", json={"key": SAVED_KEY})
    removed = client.delete("/api/providers/OpenAI/key")
    assert by_name(shell.json())["OpenAI"]["key"] == "shell"
    assert by_name(removed.json())["OpenAI"]["key"] == "shell"
    assert SHELL_KEY not in shell.text and SHELL_KEY not in removed.text


@pytest.mark.parametrize(
    ("body", "sent", "status"),
    [
        ({"key": "sk two words"}, "sk two words", 400),
        ({"key": ["sk-in-a-list"]}, "sk-in-a-list", 422),
        ({"key": 31415926535}, "31415926535", 422),
    ],
)
def test_a_malformed_key_is_refused_without_being_repeated(
    client, home, body, sent, status
):
    response = client.put("/api/providers/OpenAI/key", json=body)
    assert response.status_code == status
    assert sent not in response.text
    assert not (home / ENV_FILE).exists()


def test_a_provider_or_server_that_is_not_there_is_refused(client):
    assert (
        client.put("/api/providers/Nobody/key", json={"key": "sk-x"}).status_code == 404
    )
    assert client.delete("/api/providers/Nobody/key").status_code == 404
    # a vendor is not a custom server: its address is the app's
    assert client.put("/api/servers/OpenAI", json={"base_url": VLLM}).status_code == 404
    assert client.delete("/api/servers/OpenAI").status_code == 404


def test_a_custom_server_is_added_changed_and_removed_with_its_key(client, home):
    server = {"name": "vllm", "base_url": VLLM, "key": "sk-vllm"}
    added = client.post("/api/servers", json=server)
    again = client.post("/api/servers", json=server)
    changed = client.put(
        "/api/servers/vllm", json={"base_url": "http://127.0.0.1:9/v1"}
    )
    refused = client.put("/api/servers/vllm", json={"base_url": "ftp://127.0.0.1/v1"})
    assert added.status_code == 201
    assert by_name(added.json())["vllm"] == {
        "name": "vllm",
        "base_url": VLLM,
        "key_env": "DEEP_REASONING_VLLM_KEY",
        "custom": True,
        "key": "set",
    }
    assert "sk-vllm" not in added.text
    assert again.status_code == 400
    assert again.json()["detail"] == "There is a provider named vllm already."
    assert by_name(changed.json())["vllm"]["base_url"] == "http://127.0.0.1:9/v1"
    assert by_name(changed.json())["vllm"]["key"] == "set"
    assert refused.status_code == 400
    removed = client.delete("/api/servers/vllm")
    assert "vllm" not in by_name(removed.json())
    assert "DEEP_REASONING_VLLM_KEY" not in (home / ENV_FILE).read_text()


@pytest.mark.anyio
async def test_models_are_listed_through_the_app_with_each_providers_key(
    library_on, connect
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        async with connect(library_on(fake)) as api:
            response = await api.get("/api/models")
    assert response.json() == [
        {
            "provider": SERVER_NAME,
            "models": ["fake-large", "fake-small"],
            "problem": None,
            "featured": ["fake-large", "fake-small"],
        }
    ]
    assert fake.listed == [SERVER_KEY]
    assert SERVER_KEY not in response.text


def test_a_listing_that_fails_says_why_in_plain_words(client, home, library):
    gone = f"http://127.0.0.1:{closed_port()}/v1"
    with FakeOpenAI(scripted(PLAN), models=None) as fake:
        add_server(home, library, "unlisted", fake.base_url, None)
        add_server(home, library, "gone", gone, None)
        listings = by_name(client.get("/api/models").json(), "provider")
    assert listings["unlisted"] == {
        "provider": "unlisted",
        "models": [],
        "problem": f"No list: {fake.base_url}/models answered 404 Not Found.",
        "featured": [],
    }
    assert listings["gone"]["models"] == []
    assert listings["gone"]["problem"].startswith(
        f"No list: {gone}/models didn't answer ("
    )


@pytest.mark.parametrize("key", ["sk-in\nhalves", "sk-in\rhalves", "sk-ünïcode"])
def test_a_key_no_header_can_carry_is_never_repeated_in_a_listing(
    client, home, library, monkeypatch, key
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        add_server(home, library, "vllm", fake.base_url, None)
        monkeypatch.setenv(server_key_env("vllm"), key)
        response = client.get("/api/models")
    assert response.json() == [
        {
            "provider": "vllm",
            "models": [],
            "problem": "No list: the vllm key can't be sent; is it one line of "
            "plain text?",
            "featured": [],
        }
    ]
    assert "sk-" not in response.text


def test_a_shell_key_ending_in_a_carriage_return_lists_without_it(
    client, home, library, monkeypatch
):
    with FakeOpenAI(scripted(PLAN)) as fake:
        add_server(home, library, "vllm", fake.base_url, None)
        monkeypatch.setenv(server_key_env("vllm"), f"{SHELL_KEY}\r")
        listing = client.get("/api/models").json()[0]
    assert (listing["models"], fake.listed) == (
        ["fake-large", "fake-small"],
        [SHELL_KEY],
    )


@pytest.mark.anyio
async def test_a_run_uses_the_chats_model_and_its_providers_key_kept_out_of_the_run(
    library_on, connect, data, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-for-runs")
    look = repl("import os", "print('ANTHROPIC_API_KEY' in os.environ)")
    plan = {"Look.": [look, repl("FinalAnswer('looked')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            model = {"provider": SERVER_NAME, "name": "fake-large"}
            picked = await api.patch(f"/api/chats/{chat}", json={"model": model})
            run = await send(api, chat, "Look.")
            events = await run_events(api, run)
    assert picked.json()["model"] == model
    assert {(call.model, call.key) for call in fake.calls} == {
        ("fake-large", SERVER_KEY)
    }
    assert (events[0]["model"], events[0]["provider"]) == ("fake-large", SERVER_NAME)
    cells = [event for event in events if event["kind"] == "cell.end"]
    assert cells[0]["output"] == "False"
    assert events[-1]["outcome"] == "answered"
    for file in (data / "runs" / run).rglob("*"):
        assert not file.is_file() or SERVER_KEY.encode() not in file.read_bytes(), file


@pytest.mark.anyio
async def test_switching_namespace_mid_chat_runs_the_next_message_there(
    imported, library_on, connect
):
    plan = {"Hi.": [repl("FinalAnswer('hello')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            first = await run_events(api, await send(api, chat, "Hi."))
            switched = await api.patch(
                f"/api/chats/{chat}", json={"namespace": "course_advisor"}
            )
            again = await run_events(api, await send(api, chat, "Hi."))
            read = (await api.get(f"/api/chats/{chat}")).json()
    assert [first[0]["namespace"], again[0]["namespace"]] == [
        "router",
        "course_advisor",
    ]
    assert switched.json()["namespace"] == read["namespace"] == "course_advisor"
    texts = [message["content"] for message in fake.calls[-1].messages]
    assert texts[-3:] == ["Hi.", folded_answer("'hello'"), "Hi."]


def test_a_namespace_or_model_the_app_does_not_have_is_refused(client):
    chat = client.post("/api/chats").json()["id"]
    nowhere = client.patch(f"/api/chats/{chat}", json={"namespace": "nowhere"})
    model = {"provider": "Nobody", "name": "x"}
    nobody = client.patch(f"/api/chats/{chat}", json={"model": model})
    blank = {"provider": "OpenAI", "name": " "}
    unnamed = client.patch(f"/api/chats/{chat}", json={"model": blank})
    assert (nowhere.status_code, nobody.status_code, unnamed.status_code) == (
        400,
        400,
        422,
    )
    assert nowhere.json()["detail"].startswith('The Library has no namespace "nowhere"')
    assert nobody.json()["detail"] == "There is no provider Nobody."
    read = client.get(f"/api/chats/{chat}").json()
    assert (read["namespace"], read["model"]) == ("router", None)


def test_a_new_chat_starts_on_the_model_picked_last_in_any_chat(
    client, home, library, monkeypatch
):
    first = client.post("/api/chats").json()
    assert (first["namespace"], first["model"]) == ("router", None)
    monkeypatch.setenv("OPENAI_API_KEY", SHELL_KEY)
    luna = {"provider": "OpenAI", "name": "gpt-6-luna"}
    assert client.post("/api/chats").json()["model"] == luna
    add_server(home, library, "vllm", VLLM, None)
    qwen = {"provider": "vllm", "name": "qwen3"}
    client.patch(f"/api/chats/{first['id']}", json={"model": qwen})
    later = client.post("/api/chats").json()
    assert later["model"] == qwen
    gpt = {"provider": "OpenAI", "name": "gpt-5.5"}
    client.patch(f"/api/chats/{later['id']}", json={"model": gpt})
    models = {chat["id"]: chat["model"] for chat in client.get("/api/chats").json()}
    assert models[first["id"]] == qwen and models[later["id"]] == gpt


def test_a_message_is_refused_until_the_chat_has_a_model_that_can_run(client):
    chat = client.post("/api/chats").json()["id"]
    unpicked = client.post(f"/api/chats/{chat}/runs", json={"prompt": "Hi."})
    luna = {"provider": "OpenAI", "name": "gpt-6-luna"}
    client.patch(f"/api/chats/{chat}", json={"model": luna})
    keyless = client.post(f"/api/chats/{chat}/runs", json={"prompt": "Hi."})
    assert (unpicked.status_code, keyless.status_code) == (400, 400)
    assert (
        unpicked.json()["detail"]
        == "Pick a model for this chat, under the message box."
    )
    assert keyless.json()["detail"] == "OpenAI has no key: add one in settings."
    assert client.get(f"/api/chats/{chat}").json()["turns"] == []


def test_a_request_from_another_sites_page_is_refused(client):
    refused = client.post("/api/chats", headers={"Origin": "https://evil.example"})
    own = client.post("/api/chats", headers={"Origin": LOCAL})
    assert (refused.status_code, own.status_code) == (403, 201)
    assert len(client.get("/api/chats").json()) == 1


WRITE_THEN_READ = {
    "Write a note.": [
        repl("import pathlib", "pathlib.Path('note.txt').write_text('from message 1')"),
        repl("FinalAnswer('noted')"),
    ],
    "Read the note.": [
        repl("print(open('note.txt').read())"),
        repl("FinalAnswer('read')"),
    ],
}


def cell_outputs(events):
    return [event["output"] for event in events if event["kind"] == "cell.end"]


def folder_of(read):
    """The folder a chat's next message works in, as the page shows it."""
    return read["folder"] or read["scratch_folder"]


@pytest.mark.anyio
async def test_a_chat_without_a_folder_works_in_its_scratch_folder_across_messages(
    library_on, connect, data
):
    with FakeOpenAI(scripted(WRITE_THEN_READ)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            before = (data / "work" / chat).exists()
            empty = (await api.get(f"/api/chats/{chat}/scratch")).json()
            first = await run_events(api, await send(api, chat, "Write a note."))
            second = await run_events(api, await send(api, chat, "Read the note."))
            read = (await api.get(f"/api/chats/{chat}")).json()
            written = (await api.get(f"/api/chats/{chat}/scratch")).json()
    scratch = data / "work" / chat
    assert not before
    assert (empty, written) == ({"files": False}, {"files": True})
    assert (read["folder"], read["scratch_folder"]) == (
        None,
        f"~/files/{data.name}/work/{chat}",
    )
    assert first[0]["folder"] == second[0]["folder"] == str(scratch)
    assert (scratch / "note.txt").read_text() == "from message 1"
    assert cell_outputs(second)[0] == "from message 1"
    assert second[-1]["outcome"] == "answered", second[-1]


@pytest.mark.anyio
async def test_a_folder_changed_mid_chat_applies_from_the_next_message(
    library_on, connect, home, data
):
    thesis, data = home / "code" / "thesis", home / "data"
    thesis.mkdir(parents=True)
    data.mkdir()
    with FakeOpenAI(scripted(WRITE_THEN_READ)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            picked = await api.patch(
                f"/api/chats/{chat}", json={"folder": "~/code/thesis/"}
            )
            first = await run_events(api, await send(api, chat, "Write a note."))
            moved = await api.patch(f"/api/chats/{chat}", json={"folder": str(data)})
            second = await run_events(api, await send(api, chat, "Read the note."))
            back = await api.patch(f"/api/chats/{chat}", json={"folder": None})
    assert picked.json()["folder"] == "~/code/thesis"
    assert moved.json()["folder"] == "~/data" and back.json()["folder"] is None
    assert (first[0]["folder"], second[0]["folder"]) == (str(thesis), str(data))
    assert (thesis / "note.txt").read_text() == "from message 1"
    assert "FileNotFoundError" in cell_outputs(second)[0]
    assert not (data / "work" / chat).exists()


@pytest.mark.anyio
async def test_a_run_in_a_folder_that_is_gone_is_refused_and_nothing_starts(
    library_on, connect, home, data
):
    thesis = home / "thesis"
    thesis.mkdir()
    with FakeOpenAI(scripted(WRITE_THEN_READ)) as fake:
        async with connect(library_on(fake)) as api:
            chat = await new_chat(api)
            await api.patch(f"/api/chats/{chat}", json={"folder": str(thesis)})
            thesis.rmdir()
            refused = await api.post(
                f"/api/chats/{chat}/runs", json={"prompt": "Write a note."}
            )
            read = (await api.get(f"/api/chats/{chat}")).json()
    assert refused.status_code == 400
    assert refused.json() == {"detail": "That folder no longer exists — pick another."}
    assert read["turns"] == [] and read["title"] == "New chat"
    assert fake.calls == [] and not (data / "runs").exists()


@pytest.mark.parametrize(
    ("folder", "problem"),
    [
        ("/no/such/folder", "There is no folder /no/such/folder."),
        ("~/notes.txt", "~/notes.txt is a file, not a folder."),
    ],
)
def test_a_folder_that_is_not_there_is_refused_and_the_chat_keeps_its_own(
    client, home, folder, problem
):
    (home / "notes.txt").write_text("not a folder")
    chat = client.post("/api/chats").json()
    refused = client.patch(f"/api/chats/{chat['id']}", json={"folder": folder})
    assert refused.status_code == 400
    assert refused.json() == {"detail": problem}
    assert client.get("/api/chats").json() == [chat]


def test_folders_are_listed_for_the_picker_and_a_missing_one_is_refused(client, home):
    (home / "code" / "thesis").mkdir(parents=True)
    (home / ".cache").mkdir()
    (home / "notes.txt").write_text("not a folder")
    listed = client.get("/api/folders")
    code = client.get("/api/folders", params={"path": "~/code"})
    missing = client.get("/api/folders", params={"path": "~/gone"})
    assert listed.json() == {
        "path": "~",
        "parent": str(home.parent),
        "folders": [
            {"name": "code", "path": "~/code"},
            {"name": ".cache", "path": "~/.cache"},
        ],
    }
    assert code.json()["folders"] == [{"name": "thesis", "path": "~/code/thesis"}]
    assert (missing.status_code, missing.json()) == (
        400,
        {"detail": "There is no folder ~/gone."},
    )


def test_new_chats_start_in_the_default_folder_and_older_chats_keep_theirs(
    client, home, library
):
    (home / "code").mkdir()
    before = client.post("/api/chats").json()
    unset = client.get("/api/default-folder").json()
    saved = client.put("/api/default-folder", json={"folder": "~/code"})
    refused = client.put("/api/default-folder", json={"folder": "~/gone"})
    after = client.post("/api/chats").json()
    cleared = client.put("/api/default-folder", json={"folder": ""})
    last = client.post("/api/chats").json()
    assert unset == {"folder": None, "missing": False}
    assert saved.json() == {"folder": "~/code", "missing": False}
    assert refused.status_code == 400
    assert default_folder(library) is None
    assert cleared.json() == {"folder": None, "missing": False}
    assert [each["folder"] for each in (before, after, last)] == [None, "~/code", None]
    listed = {each["id"]: each["folder"] for each in client.get("/api/chats").json()}
    assert listed[before["id"]] is None and listed[after["id"]] == "~/code"


def test_a_default_folder_that_is_not_here_gives_new_chats_their_scratch_folder(
    client, home, data
):
    (home / "code").mkdir()
    client.put("/api/default-folder", json={"folder": "~/code"})
    (home / "code").rmdir()
    chat = client.post("/api/chats").json()
    assert (chat["folder"], chat["scratch_folder"]) == (
        None,
        f"~/files/{data.name}/work/{chat['id']}",
    )
    assert client.get("/api/default-folder").json() == {
        "folder": "~/code",
        "missing": True,
    }


@pytest.mark.anyio
async def test_deleting_a_chat_removes_its_scratch_folder_and_never_a_chosen_one(
    library_on, connect, home, data
):
    chosen = home / "thesis"
    chosen.mkdir()
    (chosen / "keep.txt").write_text("mine")
    with FakeOpenAI(scripted(WRITE_THEN_READ)) as fake:
        async with connect(library_on(fake)) as api:
            scratch, kept = await new_chat(api), await new_chat(api)
            await api.patch(f"/api/chats/{kept}", json={"folder": str(chosen)})
            await run_events(api, await send(api, scratch, "Write a note."))
            await run_events(api, await send(api, kept, "Write a note."))
            written = (data / "work" / scratch / "note.txt").is_file()
            gone = await api.delete(f"/api/chats/{scratch}")
            also = await api.delete(f"/api/chats/{kept}")
    assert written and (gone.status_code, also.status_code) == (204, 204)
    assert not (data / "work" / scratch).exists()
    assert (chosen / "keep.txt").read_text() == "mine"
    assert (chosen / "note.txt").read_text() == "from message 1"


@pytest.mark.parametrize("tampered", ["precious", "work/{chat}"])
def test_a_folder_in_a_tampered_app_file_is_treated_as_chosen_and_kept(
    client, home, library, tampered
):
    chat = client.post("/api/chats").json()["id"]
    folder = home / tampered.format(chat=chat)
    folder.mkdir(parents=True)
    (folder / "keep.txt").write_text("mine")
    with closing(open_library(library)) as connection:
        connection.execute("UPDATE chats SET folder = ?", (str(folder),))
    deleted = client.delete(f"/api/chats/{chat}")
    assert deleted.status_code == 204
    assert (folder / "keep.txt").read_text() == "mine"


def test_a_run_in_a_tampered_app_file_never_names_a_folder_to_delete(
    client, home, library, data
):
    chat = client.post("/api/chats").json()["id"]
    outside = home / "thesis"
    outside.mkdir()
    (outside / "keep.txt").write_text("mine")
    tampered = str(Path("..", "..", "..", "thesis"))
    started = {"seq": 1, "t": 100.0, "kind": "turn.start", "chat": chat}
    with closing(open_library(library)) as connection:
        save_event(connection, tampered, started)
    assert (data / "runs" / tampered).resolve() == outside
    read = client.get(f"/api/chats/{chat}").json()
    deleted = client.delete(f"/api/chats/{chat}")
    assert read["turns"] == [] and deleted.status_code == 204
    assert (outside / "keep.txt").read_text() == "mine"
    assert saved_runs(library) == set()


def test_the_default_folder_is_kept_when_it_is_in_a_deleted_chats_scratch_folder(
    client, data
):
    chat = client.post("/api/chats").json()["id"]
    inside = data / "work" / chat / "data"
    inside.mkdir(parents=True)
    (inside / "keep.txt").write_text("mine")
    client.put("/api/default-folder", json={"folder": str(inside)})
    deleted = client.delete(f"/api/chats/{chat}")
    assert deleted.status_code == 204
    assert (inside / "keep.txt").read_text() == "mine"


def plant_row(library, table, row):
    """*row* in the app file's *table*, as it is, as a file from someone else might hold."""
    marks = ", ".join("?" * len(row))
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute(f"INSERT OR REPLACE INTO {table} VALUES ({marks})", row)


def past_the_limit(start):
    """YAML starting *start*, a character longer than a Library row may be."""
    return start + "A" * (MAX_YAML_CHARS + 1 - len(start))


NOT_AN_INTEGER = "max_iter: Input should be a valid integer"
# Rows a shared file may hold that no version of the app writes, as (table, row, why the
# panel says it is left out).
UNREADABLE_ROWS = {
    "a-decomposition-not-yaml": (
        "decompositions",
        (ROOT, "bad", "name: bad\nmessages: [unclosed"),
        "The YAML doesn't parse",
    ),
    "a-decomposition-not-a-mapping": (
        "decompositions",
        (ROOT, "list", "- just\n- a\n- list\n"),
        "The YAML should be a mapping: a name: line",
    ),
    "a-decomposition-not-the-engines": (
        "decompositions",
        (ROOT, "empty", "name: empty\nmessages: []\n"),
        "messages should be a list of at least one turn",
    ),
    "a-decomposition-too-long": (
        "decompositions",
        (ROOT, "huge", past_the_limit("name: huge\nmessages: [{role: user, content: ")),
        TOO_LONG_ROW,
    ),
    "a-decomposition-whose-yaml-isnt-text": (
        "decompositions",
        (ROOT, "bytes", b"name: bytes\nmessages: [{role: user, content: hi}]\n"),
        YAML_NOT_TEXT,
    ),
    "a-namespace-the-engine-refuses": (
        "namespaces",
        ("coder", "reasoner: claude_code\n"),
        "reasoner should be a mapping",
    ),
    "a-namespace-under-none-there": (
        "namespaces",
        ("gone.child", ""),
        "is not in the Library.",
    ),
    "a-namespace-too-long": (
        "namespaces",
        ("long", past_the_limit("system_suffix: ")),
        TOO_LONG_ROW,
    ),
    "a-run-setting-the-engine-refuses": (
        "settings",
        ("max_iter", "not-a-number\n"),
        NOT_AN_INTEGER,
    ),
    "a-run-setting-the-app-never-holds": (
        "settings",
        ("reasoner", "{type: claude_code}\n"),
        "The Library never holds it",
    ),
    "a-run-setting-nested-too-deep": (
        "settings",
        ("note", "[" * 5000 + "]" * 5000),
        "The YAML is nested too deeply to read.",
    ),
    "an-entry-namespace-the-engine-doesnt-read": (
        "settings",
        ("entry_namespace", "gone\n"),
        'It names namespace \\"gone\\", which the engine doesn\'t read.',
    ),
    "a-tool-not-a-mapping": (
        "tools",
        ("tally", "- factory: build\n", None),
        "tools.tally: Input should be a valid dictionary",
    ),
    "a-tool-whose-description-isnt-text": (
        "tools",
        ("tally", "factory: llm\ndescription: !!binary /w==\n", None),
        "Its description should be text.",
    ),
    "a-tool-whose-source-is-too-long": (
        "tools",
        ("tally", "factory: build\nfactory_from: tally.py\n", past_the_limit("#")),
        TOO_LONG_ROW,
    ),
    "a-tool-of-no-python-naming-no-built-in-factory": (
        "tools",
        ("tally", "factory: build\n", None),
        'names factory \\"build\\", which is not built in',
    ),
    "a-tool-of-python-naming-no-file": (
        "tools",
        ("tally", "factory: llm\n", WORD_COUNT),
        "A tool holds Python exactly when its block names a file for it",
    ),
    "a-tool-whose-factory-isnt-text": (
        "tools",
        ("tally", "factory: [build]\nfactory_from: tally.py\n", WORD_COUNT),
        "factory and factory_from should be text.",
    ),
    "a-tool-of-python-named-as-claude-code": (
        "tools",
        ("tally", "factory: claude_code\nfactory_from: tally.py\n", WORD_COUNT),
        "one of the engine's own",
    ),
}
# What the panel deletes a row of each table with, keyed as the row.
DELETES = {
    "decompositions": "/api/decompositions",
    "namespaces": "/api/namespaces/{}",
    "settings": "/api/run-settings/{}",
    "tools": "/api/tools/{}",
}


def deleting(table, row):
    """The URL and arguments the panel deletes *row* of *table* with: a decomposition by
    the YAML it was shown, null for one too long to show or not text."""
    url = DELETES[table].format(row[0])
    if table == "decompositions":
        namespace, name, text = row
        shown = type(text) is str and len(text) <= MAX_YAML_CHARS
        body = {"namespace": namespace, "name": name, "was": text if shown else None}
        return url, {"json": body}
    if table == "namespaces":
        return url, {"params": {"decompositions": 0}}
    return url, {}


def rows_keyed(library, table, row):
    """How many rows of *table* hold *row*'s key."""
    key = {"decompositions": "namespace = ? AND name = ?", "settings": "key = ?"}
    where = key.get(table, "name = ?")
    width = where.count("?")
    with closing(open_library(library)) as connection:
        query = f"SELECT count(*) FROM {table} WHERE {where}"
        return connection.execute(query, row[:width]).fetchone()[0]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("table", "row", "why"), UNREADABLE_ROWS.values(), ids=UNREADABLE_ROWS.keys()
)
async def test_a_shared_files_row_the_engine_cant_read_is_left_out_saying_why(
    library_on, connect, table, row, why
):
    plan = {"Hi.": [repl("FinalAnswer('hello')")]}
    with FakeOpenAI(scripted(plan)) as fake:
        library = library_on(fake)
        plant_row(library, table, row)
        with capture_logs() as logs:
            async with connect(library) as api:
                shown = await api.get("/api/library")
                chat = await new_chat(api)
                events = await run_events(api, await send(api, chat, "Hi."))
                unrelated = await api.post(
                    "/api/namespaces", json={"name": "notes", "yaml": ""}
                )
                url, arguments = deleting(table, row)
                deleted = await api.request("DELETE", url, **arguments)
    key = row[:2] if table == "decompositions" else row[0]
    left_out = {
        (entry["table"], entry["row"])
        for entry in logs
        if entry["event"] == "library.row_left_out"
    }
    assert shown.status_code == 200 and why in shown.text
    # A row past the limit is never loaded, so never sent.
    assert len(shown.text) < MAX_YAML_CHARS
    assert left_out == {(table, repr(key))}
    assert events[-1]["outcome"] == "answered"
    assert unrelated.status_code == 201, unrelated.text
    assert deleted.status_code == 200, deleted.text
    assert rows_keyed(library, table, row) == 0


HELLO = {"Hi.": [repl("FinalAnswer('hello')")]}


@pytest.mark.anyio
async def test_a_granted_tool_left_out_is_ignored_so_runs_and_writes_go_on(
    library_on, connect
):
    with FakeOpenAI(scripted(HELLO)) as fake:
        library = library_on(fake)
        plant_row(library, "tools", ("tally", "- not a mapping\n", None))
        plant_row(library, "namespaces", ("drafts", "tools: [tally]\n"))
        async with connect(library) as api:
            chat = await new_chat(api)
            await api.patch(f"/api/chats/{chat}", json={"namespace": "drafts"})
            events = await run_events(api, await send(api, chat, "Hi."))
            unrelated = await api.post(
                "/api/namespaces", json={"name": "notes", "yaml": ""}
            )
            refused = await api.delete("/api/tools/tally")
            tally = by_name((await api.get("/api/library")).json()["tools"])["tally"]
    assert events[-1]["outcome"] == "answered"
    assert unrelated.status_code == 201, unrelated.text
    assert (tally["granted_in"], tally["problems"][-1]) == (["drafts"], GRANT_IGNORED)
    assert refused.json()["detail"] == tally["refusal"]
    assert tally["refusal"] == (
        "tally is granted in drafts: take it out of their tools first."
    )


def test_an_entry_namespace_naming_one_left_out_reads_as_root_and_can_be_deleted(
    client, library
):
    plant_row(library, "namespaces", ("coder", "reasoner: claude_code\n"))
    plant_row(library, "settings", ("entry_namespace", "coder\n"))
    shown = client.get("/api/library").json()
    unrelated = client.post("/api/namespaces", json={"name": "notes", "yaml": ""})
    deleted = client.delete("/api/run-settings/entry_namespace")
    [entry] = by_name(shown["run_settings"], "label")["entry_namespace"]["items"]
    assert (shown["default_namespace"], entry["value"]) == (ROOT, ROOT)
    assert (
        'It names namespace "coder", which the engine doesn\'t read.' in entry["source"]
    )
    assert (unrelated.status_code, deleted.status_code) == (201, 200)


@pytest.mark.anyio
async def test_a_left_out_root_runs_bare_and_an_edit_from_a_blank_draft_replaces_it(
    library_on, connect
):
    with FakeOpenAI(scripted(HELLO)) as fake:
        library = library_on(fake)
        plant_row(library, "namespaces", (ROOT, past_the_limit("system_suffix: ")))
        async with connect(library) as api:
            [root, *_] = (await api.get("/api/library")).json()["namespaces"]
            chat = await new_chat(api)
            events = await run_events(api, await send(api, chat, "Hi."))
            body = {"yaml": "tools: [llm]\n", "was": None}
            replaced = await api.put(f"/api/namespaces/{ROOT}", json=body)
            again = await api.put(f"/api/namespaces/{ROOT}", json=body)
    assert (root["yaml"], root["usable"], root["problems"]) == (
        None,
        True,
        [TOO_LONG_ROW],
    )
    assert events[-1]["outcome"] == "answered"
    [now, *_] = replaced.json()["namespaces"]
    assert (now["yaml"], now["problems"]) == ("tools: [llm]\n", [])
    # No longer too long to show, so a draft opened on that replaces nothing.
    assert again.status_code == 409


def test_a_namespace_saved_too_long_is_refused_saying_so(client):
    text = past_the_limit("system_suffix: ")
    refused = client.post("/api/namespaces", json={"name": "long", "yaml": text})
    assert (refused.status_code, refused.json()["detail"]) == (
        400,
        "A namespace's YAML may be at most 1,000,000 characters; this has 1,000,001.",
    )


def test_a_decomposition_the_engine_cant_read_shows_its_yaml_and_why_agents_dont_see_it(
    client, library
):
    plant_row(library, "decompositions", (ROOT, "list", "- a\n"))
    shown = client.get("/api/library").json()
    [root] = [each for each in shown["namespaces"] if each["name"] == ROOT]
    assert by_name(root["decompositions"])["list"] == {
        "name": "list",
        "yaml": "- a\n",
        "problems": [
            UNREAD_DECOMPOSITION,
            (
                "The YAML should be a mapping: a name: line, then messages:, as in the"
                " template."
            ),
        ],
    }


def test_malformed_settings_read_as_unset_and_the_page_can_set_them_again(
    client, library
):
    planted = [(SERVERS, '"x"'), (LAST_MODEL, "1"), (DEFAULT_FOLDER, '"C:/Users/me"')]
    with closing(open_library(library)) as connection:
        connection.executemany("INSERT INTO app_settings VALUES (?, ?)", planted)
    chat = client.post("/api/chats")
    folder = client.get("/api/default-folder").json()
    added = client.post("/api/servers", json={"name": "vllm", "base_url": VLLM})
    qwen = {"provider": "vllm", "name": "qwen3"}
    client.patch(f"/api/chats/{chat.json()['id']}", json={"model": qwen})
    assert (chat.status_code, chat.json()["model"], folder) == (
        201,
        None,
        {"folder": None, "missing": False},
    )
    assert (
        added.status_code == 201 and by_name(added.json())["vllm"]["base_url"] == VLLM
    )
    assert client.post("/api/chats").json()["model"] == qwen


@pytest.mark.anyio
async def test_a_folder_chosen_inside_a_deleted_chats_run_folder_is_kept(
    library_on, connect, data
):
    with FakeOpenAI(scripted(WRITE_THEN_READ)) as fake:
        async with connect(library_on(fake)) as api:
            gone, kept = await new_chat(api), await new_chat(api)
            run = await send(api, gone, "Write a note.")
            await run_events(api, run)
            inside = data / "runs" / run / "mine"
            inside.mkdir()
            (inside / "keep.txt").write_text("mine")
            await api.patch(f"/api/chats/{kept}", json={"folder": str(inside)})
            deleted = await api.delete(f"/api/chats/{gone}")
    assert deleted.status_code == 204
    assert (inside / "keep.txt").read_text() == "mine"


@pytest.mark.anyio
async def test_a_folder_another_chats_run_works_in_is_kept_after_that_chat_moved_on(
    library_on, connect, data
):
    held, release = held_on("Count to two.", PLAN)
    with FakeOpenAI(held) as fake:
        async with connect(library_on(fake)) as api:
            gone, working = await new_chat(api), await new_chat(api)
            scratch = data / "work" / gone
            scratch.mkdir(parents=True)
            (scratch / "keep.txt").write_text("mine")
            await api.patch(f"/api/chats/{working}", json={"folder": str(scratch)})
            run = await send(api, working, "Count to two.")
            await api.patch(f"/api/chats/{working}", json={"folder": None})
            deleted = await api.delete(f"/api/chats/{gone}")
            release.set()
            events = await run_events(api, run)
    assert deleted.status_code == 204
    assert events[0]["folder"] == str(scratch)
    assert (scratch / "keep.txt").read_text() == "mine"
