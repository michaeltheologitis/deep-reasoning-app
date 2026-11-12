import json
import re
import time
from contextlib import closing
from itertools import count

import pytest
from deep_reasoner.namespaces import ROOT
from structlog.testing import capture_logs

from deep_reasoning import chats as chats_module
from deep_reasoning.chats import (
    CHAT_ID,
    ID_TRIES,
    NEW_TITLE,
    choose,
    choose_folder,
    create_chat,
    delete_chat,
    find_chat,
    list_chats,
    rename_chat,
    title_new_chat,
)
from deep_reasoning.library import (
    ADDED_IN,
    BEFORE_CHANGES,
    BEFORE_EVENTS,
    BEFORE_PAGES,
    HISTORIAN,
    LIBRARIAN,
    LIBRARY_ONLY,
    SCHEMA_VERSION,
    LibraryError,
    add_namespace,
    default_namespace,
    open_library,
    read_changes,
    schema_version,
    stored_config,
    writing,
)
from deep_reasoning.pages import add_page, read_page
from deep_reasoning.runs import chat_runs, read_events, save_event
from deep_reasoning.settings import (
    LAST_MODEL,
    read_settings,
    remember_pick,
    save_default_folder,
)

MANY = 200
NOW = 1_800_000_000.0
ADVISORS_NAMESPACES = {"root", "router", "course_advisor", "health_advisor"}


def as_made_by(version, library):
    """The file as *version* made it, without the tables of the versions after it."""
    later = [
        table
        for added, (tables, _) in ADDED_IN.items()
        if added > version
        for table in tables
    ]
    with closing(open_library(library)) as connection:
        for table in later:
            connection.execute(f"DROP TABLE {table}")
        connection.execute(f"PRAGMA user_version = {version}")


def test_a_chat_made_first_in_a_new_file_leaves_a_whole_library_beside_it(library):
    chat = create_chat(library)
    assert list_chats(library) == [chat]
    assert default_namespace(library) == ROOT


def test_a_version_1_library_opens_with_its_library_kept_and_gains_chats(imported):
    with closing(open_library(imported)) as connection:
        # Made before a fresh Library had a historian or librarian: opening it adds neither.
        for name in (HISTORIAN, LIBRARIAN):
            connection.execute("DELETE FROM namespaces WHERE name = ?", (name,))
            connection.execute(
                "DELETE FROM decompositions WHERE namespace = ?", (name,)
            )
        before = stored_config(connection)
    as_made_by(LIBRARY_ONLY, imported)
    chat = create_chat(imported)
    with closing(open_library(imported)) as connection:
        after = stored_config(connection)
        version = schema_version(connection)
    assert after == before and set(after["namespaces"]) == ADVISORS_NAMESPACES
    assert list_chats(imported) == [chat]
    assert version == SCHEMA_VERSION


def test_a_version_2_file_opens_with_its_chats_and_settings_kept_and_gains_runs(
    imported, tmp_path
):
    chat = create_chat(imported, str(tmp_path))
    rename_chat(imported, chat["id"], "Plans")
    remember_pick(imported, "OpenAI", "gpt-6-luna")
    save_default_folder(imported, str(tmp_path))
    before, settings = list_chats(imported), read_settings(imported)
    as_made_by(BEFORE_EVENTS, imported)
    run = "20261025-120000-000000"
    started = {"seq": 1, "t": 100.0, "kind": "turn.start", "chat": chat["id"]}
    with closing(open_library(imported)) as connection:
        version = schema_version(connection)
        save_event(connection, run, started)
        runs = chat_runs(connection, chat["id"])
        events = read_events(connection, run)
    assert version == SCHEMA_VERSION
    assert list_chats(imported) == [before[0] | {"last_run": run}]
    assert before[0]["title"] == "Plans"
    assert read_settings(imported) == settings
    assert settings[LAST_MODEL] == {"provider": "OpenAI", "name": "gpt-6-luna"}
    assert (runs, events) == ([run], [started])


def test_a_version_3_file_opens_with_all_it_holds_kept_and_gains_pages(
    imported, tmp_path
):
    chat = create_chat(imported, str(tmp_path))["id"]
    remember_pick(imported, "OpenAI", "gpt-6-luna")
    run = "20261025-120000-000000"
    started = {"seq": 1, "t": 100.0, "kind": "turn.start", "chat": chat}
    with closing(open_library(imported)) as connection:
        save_event(connection, run, started)
        library = stored_config(connection)
    chats, settings = list_chats(imported), read_settings(imported)
    as_made_by(BEFORE_PAGES, imported)
    blocks = [{"type": "text", "text": "Kept."}]
    with closing(open_library(imported)) as connection:
        version = schema_version(connection)
        kept = (stored_config(connection), chat_runs(connection, chat))
        events = read_events(connection, run)
        added = add_page(connection, chat, run, 1, "New", json.dumps(blocks))
        page = read_page(connection, added)
    assert version == SCHEMA_VERSION
    assert (list_chats(imported), read_settings(imported)) == (chats, settings)
    assert kept == (library, [run])
    assert events == [started]
    assert page["blocks"] == blocks


def test_a_version_4_file_opens_with_all_it_holds_kept_and_gains_the_record_of_changes(
    imported, tmp_path
):
    chat = create_chat(imported, str(tmp_path))["id"]
    run = "20261025-120000-000000"
    started = {"seq": 1, "t": 100.0, "kind": "turn.start", "chat": chat}
    blocks = [{"type": "text", "text": "Kept."}]
    with closing(open_library(imported)) as connection:
        save_event(connection, run, started)
        page = add_page(connection, chat, run, 1, "Kept", json.dumps(blocks))
        library = stored_config(connection)
    chats = list_chats(imported)
    as_made_by(BEFORE_CHANGES, imported)
    with closing(open_library(imported)) as connection:
        version = schema_version(connection)
        kept = (stored_config(connection), read_events(connection, run))
        kept_page = read_page(connection, page)
        add_namespace(connection, "notes", "")
        recorded = read_changes(connection)
    assert version == SCHEMA_VERSION
    assert (kept, list_chats(imported)) == ((library, [started]), chats)
    assert kept_page["blocks"] == blocks
    assert [(each["what"], each["name"], each["by"]) for each in recorded] == [
        ("namespace", "notes", "panel")
    ]


def test_new_chats_are_titled_new_chat_and_listed_newest_first(library):
    first, second = create_chat(library), create_chat(library)
    assert first["title"] == second["title"] == NEW_TITLE
    assert list_chats(library) == [second, first]


def test_chats_made_in_the_same_clock_tick_are_listed_newest_first(
    library, monkeypatch
):
    monkeypatch.setattr(time, "time", lambda: NOW)
    first, second = create_chat(library), create_chat(library)
    assert list_chats(library) == [second, first]


def test_chats_are_listed_by_their_latest_run_else_when_made(library, monkeypatch):
    clock = count(NOW)
    monkeypatch.setattr(time, "time", lambda: next(clock))
    used, older, newer = (
        create_chat(library),
        create_chat(library),
        create_chat(library),
    )
    run = "20330101-000000-000000"  # after all three were made
    with closing(open_library(library)) as connection:
        connection.execute("INSERT INTO runs VALUES (?, ?)", (run, used["id"]))
    listed = [(chat["id"], chat["last_run"]) for chat in list_chats(library)]
    assert listed == [(used["id"], run), (newer["id"], None), (older["id"], None)]


def test_a_run_id_the_app_could_not_have_made_is_ignored(library, monkeypatch):
    clock = count(NOW)
    monkeypatch.setattr(time, "time", lambda: next(clock))
    dateless, shapeless = create_chat(library), create_chat(library)
    with closing(open_library(library)) as connection:
        for run, chat in [
            ("99999999-not-a-run", shapeless),
            (b"20330101-000000-000000", shapeless),
            # its shape, so chat_runs lists it, but month 13: no date to order it by
            ("20331301-000000-000000", dateless),
        ]:
            connection.execute("INSERT INTO runs VALUES (?, ?)", (run, chat["id"]))
    listed = [(chat["id"], chat["last_run"]) for chat in list_chats(library)]
    assert listed == [
        (shapeless["id"], None),
        (dateless["id"], "20331301-000000-000000"),
    ]
    assert find_chat(library, shapeless["id"])["last_run"] is None


def test_ids_are_readable_url_safe_and_unique(library):
    ids = [create_chat(library)["id"] for _ in range(MANY)]
    assert len(set(ids)) == MANY
    assert all(re.fullmatch(CHAT_ID, chat) for chat in ids)
    assert all(re.fullmatch(r"[a-z]+(-[a-z]+)+", chat) for chat in ids)


def test_an_id_already_taken_is_drawn_again(library, monkeypatch):
    drawn = iter(["quiet-blue-heron", "quiet-blue-heron", "brave-red-fox"])
    monkeypatch.setattr(chats_module, "generate_slug", lambda words: next(drawn))
    first, second = create_chat(library), create_chat(library)
    assert (first["id"], second["id"]) == ("quiet-blue-heron", "brave-red-fox")
    assert list_chats(library) == [second, first]


def test_a_chat_whose_every_id_tried_is_taken_is_refused_in_words(library, monkeypatch):
    taken = create_chat(library)
    monkeypatch.setattr(chats_module, "generate_slug", lambda words: taken["id"])
    with pytest.raises(LibraryError, match=f"no free chat id in {ID_TRIES} tries"):
        create_chat(library)
    assert list_chats(library) == [taken]


def test_a_shared_files_chats_table_that_refuses_every_row_refuses_a_new_chat_in_words(
    library,
):
    with closing(open_library(library)) as connection, writing(connection):
        connection.execute("DROP TABLE chats")
        connection.execute(
            "CREATE TABLE chats (id TEXT PRIMARY KEY, title TEXT NOT NULL,"
            " created REAL NOT NULL, namespace TEXT, provider TEXT, model TEXT,"
            " folder TEXT, CHECK (0))"
        )
    with pytest.raises(
        LibraryError, match="refused the new chat: CHECK constraint failed"
    ):
        create_chat(library)


def test_rename_changes_only_that_chats_title(library):
    kept, renamed = create_chat(library), create_chat(library)
    rename_chat(library, renamed["id"], "Trip to Rome")
    assert find_chat(library, renamed["id"]) == renamed | {"title": "Trip to Rome"}
    assert find_chat(library, kept["id"]) == kept


@pytest.mark.parametrize(
    ("message", "title"),
    [
        ("Plan a trip.\nTo Rome, in May.", "Plan a trip."),
        ("\n\n  Plan a trip.  \r\nTo Rome.", "Plan a trip."),
        (" \n\t\n", NEW_TITLE),
        ("Plan a trip.\u2028To Rome.\nIn May.", "Plan a trip."),
    ],
)
def test_a_new_chat_is_titled_after_its_first_messages_first_line(
    library, message, title
):
    chat = create_chat(library)
    title_new_chat(library, chat["id"], message)
    assert find_chat(library, chat["id"])["title"] == title


@pytest.mark.parametrize("typed", ["Trip\nto Rome", "Trip\r\nto Rome", "Trip\vto Rome"])
def test_a_rename_holding_a_line_break_puts_the_title_on_one_line(library, typed):
    chat = create_chat(library)
    rename_chat(library, chat["id"], typed)
    assert find_chat(library, chat["id"])["title"] == "Trip to Rome"


def test_a_renamed_chat_is_not_retitled(library):
    chat = create_chat(library)
    rename_chat(library, chat["id"], "Trip to Rome")
    title_new_chat(library, chat["id"], "Plan a trip.")
    assert find_chat(library, chat["id"])["title"] == "Trip to Rome"


def test_a_deleted_chat_is_gone_and_the_others_stay(library):
    gone, kept = create_chat(library), create_chat(library)
    delete_chat(library, gone["id"])
    assert find_chat(library, gone["id"]) is None
    assert list_chats(library) == [kept]


def test_a_new_chat_has_picked_nothing_yet(library):
    chat = create_chat(library)
    picks = (chat["namespace"], chat["provider"], chat["model"], chat["folder"])
    assert picks == (None, None, None, None)
    assert find_chat(library, chat["id"]) == chat


def test_a_new_chat_starts_in_the_folder_it_is_given(library):
    chat = create_chat(library, "/home/me/code/thesis")
    assert chat["folder"] == find_chat(library, chat["id"])["folder"]
    assert chat["folder"] == "/home/me/code/thesis"


def test_a_folder_choice_changes_only_that_chat_and_can_go_back_to_scratch(library):
    chat, other = create_chat(library), create_chat(library)
    choose_folder(library, chat["id"], "/home/me/code/thesis")
    choose(library, chat["id"], namespace="router")
    chosen = find_chat(library, chat["id"])["folder"]
    choose_folder(library, chat["id"], None)
    assert chosen == "/home/me/code/thesis"
    assert find_chat(library, chat["id"])["folder"] is None
    assert find_chat(library, other["id"]) == other


def test_a_choice_changes_only_what_it_names_and_only_that_chat(library):
    chat, other = create_chat(library), create_chat(library)
    choose(library, chat["id"], "router", "OpenAI", "gpt-6-luna")
    choose(library, chat["id"], namespace="course_advisor")
    choose(library, chat["id"], provider="vllm", model="qwen3")
    found = find_chat(library, chat["id"])
    assert (found["namespace"], found["provider"], found["model"]) == (
        "course_advisor",
        "vllm",
        "qwen3",
    )
    assert find_chat(library, other["id"]) == other


@pytest.mark.parametrize(
    ("planted", "folder"),
    [
        ("../x", None),
        ("x" * 101, None),
        (b"not text", None),
        ("planted-chat", "relative/folder"),
        ("planted-chat", "C:/Users/me"),
        ("planted-chat", b"/not/text"),
    ],
)
def test_a_chat_row_the_app_could_not_have_written_is_left_out(
    library, planted, folder
):
    chat = create_chat(library)
    with closing(open_library(library)) as connection:
        connection.execute(
            "INSERT INTO chats (id, title, created, folder) VALUES (?, 'Planted', 1.0, ?)",
            (planted, folder),
        )
    with capture_logs() as logs:
        listed = list_chats(library)
        found = find_chat(library, "planted-chat")
    assert listed == [chat] and found is None
    assert "chats.left_out" in {entry["event"] for entry in logs}
