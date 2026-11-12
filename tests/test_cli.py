import errno
import fcntl
import io
import json
import os
import queue
import re
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from contextlib import closing
from pathlib import Path

import pytest
import structlog
import uvicorn
from conftest import CONFIGS
from fastapi.testclient import TestClient

from deep_reasoning import cli
from deep_reasoning.chats import create_chat
from deep_reasoning.folders import data_folder
from deep_reasoning.library import ADDED_IN, open_library
from deep_reasoning.settings import NOT_A_KEY, VENDORS

WAIT_SECONDS = 0.5
KEY_IN_A_FRAME = "sk-held-in-a-frame"
KEYS = "OPENAI_API_KEY=sk-kept-by-reset\n"
# A process that holds the lock beside the Library it is given, as an app serving it does,
# until it is killed.
HOLD_THE_LOCK = """
import sys, time
from pathlib import Path
from deep_reasoning import cli
with cli.owning(Path(sys.argv[1]), "held"):
    print("held", flush=True)
    time.sleep(600)
"""


@pytest.fixture
def free_port():
    with socket.create_server(("127.0.0.1", 0)) as probe:
        return probe.getsockname()[1]


@pytest.fixture
def opened(monkeypatch):
    """The URLs the browser was asked to open; each must already accept connections."""
    urls = queue.Queue()

    def open_tab(url):
        address = urllib.parse.urlsplit(url)
        socket.create_connection((address.hostname, address.port)).close()
        urls.put(url)

    monkeypatch.setattr(webbrowser, "open", open_tab)
    return urls


@pytest.fixture
def served(monkeypatch):
    """The addresses uvicorn served on. The fake stops at once."""
    addresses = []

    def serve(server, sockets):
        addresses.append(sockets[0].getsockname())

    monkeypatch.setattr(uvicorn.Server, "run", serve)
    return addresses


def test_serves_on_localhost_only(served, free_port):
    cli.main(["--port", str(free_port), "--no-browser"])
    assert served == [("127.0.0.1", free_port)]


def test_says_where_it_runs_and_which_library_it_serves(
    served, capsys, home, free_port
):
    cli.main(["--port", str(free_port), "--no-browser"])
    said = capsys.readouterr().out
    assert f"url=http://127.0.0.1:{free_port}/" in said
    assert str(home / ".deep-reasoning" / "library.sqlite") in said


def test_a_traceback_the_server_logs_shows_no_frames_variables(
    served, capsys, free_port
):
    cli.main(["--port", str(free_port), "--no-browser"])

    def refused(key):
        raise RuntimeError("the provider refused")

    try:
        refused(KEY_IN_A_FRAME)
    except RuntimeError:
        structlog.get_logger().exception("models.not_listed")
    said = capsys.readouterr().out
    assert "RuntimeError: the provider refused" in said
    assert KEY_IN_A_FRAME not in said


def test_opens_a_tab_at_the_port_it_listens_on(monkeypatch, opened):
    tabs = []

    def serve(server, sockets):
        port = sockets[0].getsockname()[1]
        tabs.append((opened.get(timeout=WAIT_SECONDS), port))

    monkeypatch.setattr(uvicorn.Server, "run", serve)
    cli.main(["--port", "0"])
    [(url, port)] = tabs
    assert url == f"http://127.0.0.1:{port}/"


def test_no_browser_opens_no_tab(monkeypatch, opened, free_port):
    def serve(server, sockets):
        with pytest.raises(queue.Empty):
            opened.get(timeout=WAIT_SECONDS)

    monkeypatch.setattr(uvicorn.Server, "run", serve)
    cli.main(["--port", str(free_port), "--no-browser"])


def test_a_taken_port_exits_with_a_plain_error_making_no_library(served, tmp_path):
    folder = tmp_path / "libraries"
    folder.mkdir()
    library = folder / "new.sqlite"
    with socket.create_server(("127.0.0.1", 0)) as taken:
        port = taken.getsockname()[1]
        with pytest.raises(SystemExit, match=f"Port {port} is already in use.*--port"):
            cli.main(["--port", str(port), "--no-browser", "--library", str(library)])
    assert served == []
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("port", ["-1", "65536"])
def test_a_port_out_of_range_exits_with_a_plain_error(served, capsys, port):
    with pytest.raises(SystemExit):
        cli.main(["--port", port, "--no-browser"])
    assert "--port must be between 0 and 65535" in capsys.readouterr().err
    assert served == []


def test_ctrl_c_stops_the_server_without_a_traceback(monkeypatch, free_port):
    def interrupted(server, sockets):
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn.Server, "run", interrupted)
    try:
        cli.main(["--port", str(free_port), "--no-browser"])
    except KeyboardInterrupt:
        pytest.fail("Ctrl+C escaped main, so the terminal shows a traceback")


def test_the_library_defaults_to_one_under_the_home_directory(served, home, free_port):
    cli.main(["--port", str(free_port), "--no-browser"])
    assert (home / ".deep-reasoning" / "library.sqlite").is_file()


def test_library_opens_the_given_file_and_import_fills_it(
    monkeypatch, tmp_path, advisors, free_port
):
    shared = tmp_path / "shared.sqlite"
    cli.main(["import", str(advisors), "--library", str(shared)])
    seen = []

    def serve(server, sockets):
        client = TestClient(server.config.app, base_url="http://127.0.0.1")
        seen.append(client.get("/api/library").json())

    monkeypatch.setattr(uvicorn.Server, "run", serve)
    cli.main(["--port", str(free_port), "--no-browser", "--library", str(shared)])
    [library] = seen
    assert library["file"] == "shared.sqlite"
    assert library["default_namespace"] == "router"


@pytest.mark.parametrize(
    ("config", "said"),
    [
        (None, "Could not import .*; the Library is unchanged"),
        (
            "max_iter: 0\n",
            "Could not import .*; the Library is unchanged.\nmax_iter is 0",
        ),
    ],
    ids=["a config that isn't there", "one it refuses"],
)
def test_a_failed_import_into_a_new_path_says_so_and_leaves_no_library_or_lock(
    tmp_path, config, said
):
    given = tmp_path / "config.yaml"
    if config:
        given.write_text(config)
    library = tmp_path / "lib.sqlite"
    with pytest.raises(SystemExit, match=said):
        cli.main(["import", str(given), "--library", str(library)])
    assert not library.exists()


def test_an_import_that_fails_in_any_way_leaves_no_new_library(tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError("a bug")

    monkeypatch.setattr(cli, "import_config", broken)
    config = tmp_path / "steps.yaml"
    config.write_text("max_iter: 4\n")
    library = tmp_path / "lib.sqlite"
    with pytest.raises(RuntimeError, match="a bug"):
        cli.main(["import", str(config), "--library", str(library)])
    assert not library.exists()


def test_a_library_made_before_the_import_took_the_lock_is_kept_when_it_fails(
    tmp_path, monkeypatch
):
    library = tmp_path / "lib.sqlite"
    flock = fcntl.flock

    def served_meanwhile(lock, operation):
        # An app served the path, and a chat was made there, as this import started.
        if not library.exists():
            create_chat(library)
        flock(lock, operation)

    monkeypatch.setattr(fcntl, "flock", served_meanwhile)
    config = tmp_path / "steps.yaml"
    config.write_text("max_iter: 0\n")
    with pytest.raises(SystemExit, match="max_iter is 0"):
        cli.main(["import", str(config), "--library", str(library)])
    with closing(open_library(library)) as connection:
        assert connection.execute("SELECT count(*) FROM chats").fetchone() == (1,)


def test_an_import_that_changes_nothing_in_a_new_library_makes_none(tmp_path, capsys):
    config = tmp_path / "nothing.yaml"
    config.write_text("namespaces: {}\n")
    library = tmp_path / "lib.sqlite"
    cli.main(["import", str(config), "--library", str(library)])
    said = (
        f"Nothing to import: a new Library already holds all of {config}, so none was"
        f" made at {library}."
    )
    assert capsys.readouterr().out.splitlines() == [said]
    assert not library.exists()


@pytest.fixture
def read_only():
    """Make a file read only, even to root (by chattr +i, skipping where it isn't
    permitted), until the test ends."""
    made = []

    def make(path):
        path.chmod(0o444)
        if not os.access(path, os.W_OK):
            return
        if shutil.which("chattr") is None:
            pytest.skip("root writes a read-only file, and there is no chattr")
        immutable = subprocess.run(
            ["chattr", "+i", path], check=False, capture_output=True
        )
        if immutable.returncode != 0:
            pytest.skip("root writes a read-only file, and chattr +i isn't permitted")
        made.append(path)

    yield make
    for path in made:
        subprocess.run(["chattr", "-i", path], check=True)


def test_an_import_into_a_read_only_library_says_so_in_plain_words(tmp_path, read_only):
    library = tmp_path / "lib.sqlite"
    open_library(library).close()
    read_only(library)
    config = tmp_path / "steps.yaml"
    config.write_text("max_iter: 4\n")
    said = (
        f"Could not import {config}; the Library is unchanged.\nThe Library file can't"
        " be written: attempt to write a readonly database."
    )
    with pytest.raises(SystemExit) as refused:
        cli.main(["import", str(config), "--library", str(library)])
    assert refused.value.code == said


# The command, as its script runs it.
COMMAND = "import sys; from deep_reasoning.cli import main; main(sys.argv[1:])"
# Builds slowly, saying first that it has begun.
SLOW_TO_BUILD = """\
import os, time
from pathlib import Path

def build(client, params):
    Path(params["started"]).write_text(str(os.getpid()))
    time.sleep(60)
"""


def test_ctrl_c_during_an_import_stops_it_saying_so_and_leaves_nothing(tmp_path):
    (tmp_path / "slow.py").write_text(SLOW_TO_BUILD)
    started = tmp_path / "started"
    config = tmp_path / "slow.yaml"
    config.write_text(
        f"tools:\n  slow: {{factory: build, factory_from: slow.py, started: {started}}}\n"
        "namespaces:\n  helper: {tools: [slow]}\n"
    )
    library = tmp_path / "lib.sqlite"
    importing = subprocess.Popen(
        [
            sys.executable,
            "-c",
            COMMAND,
            "import",
            str(config),
            "--library",
            str(library),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while not started.exists():
            assert importing.poll() is None
            time.sleep(WAIT_SECONDS)
        importing.send_signal(signal.SIGINT)
        _, said = importing.communicate(timeout=30)
    finally:
        importing.kill()
        importing.wait()
    assert said.strip().splitlines()[-1] == cli.STOPPED
    assert "Traceback" not in said
    assert not library.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(started.read_text()), 0)


def test_import_says_what_it_changed(tmp_path, capsys, waitlist):
    library = tmp_path / "lib.sqlite"
    cli.main(
        ["import", str(waitlist), "--within", str(CONFIGS), "--library", str(library)]
    )
    said = capsys.readouterr().out.splitlines()
    header = said.index(f"Imported {waitlist} into the Library at {library}.")
    assert "  root: tools [llm] → [waitlist]; the rest of root kept" in said[header:]


def test_an_imports_report_alone_is_on_stdout_its_log_lines_on_stderr(tmp_path, capsys):
    library = tmp_path / "lib.sqlite"
    open_library(library).close()
    with closing(open_library(library)) as connection, connection:
        # A row a shared file may hold, which every read of the Library warns of.
        connection.execute("INSERT INTO namespaces VALUES ('Root', '')")
    config = tmp_path / "config.yaml"
    config.write_text("max_iter: 7\n")
    cli.main(["import", str(config), "--library", str(library)])
    said = capsys.readouterr()
    assert said.out.splitlines() == [
        f"Imported {config} into the Library at {library}.",
        "Run settings, for every chat:",
        "  max_iter: 30 (the engine's default) → 7",
    ]
    assert "library.row_left_out" in said.err


def test_importing_a_config_again_says_it_changed_nothing(tmp_path, capsys):
    config = tmp_path / "steps.yaml"
    config.write_text("max_iter: 4\n")
    library = tmp_path / "lib.sqlite"
    for _ in range(2):
        cli.main(["import", str(config), "--library", str(library)])
    assert capsys.readouterr().out.splitlines()[-1] == (
        f"Nothing to import: the Library at {library} already holds all of {config}."
    )


# Reads a proxy and a provider's key as it is built, as a tool may in every run.
READS_THE_ENVIRONMENT = """\
import json, os
from pathlib import Path
from deep_reasoner.primitives import Func

PROXY = os.environ["HTTPS_PROXY"]
KEY = os.environ["OPENAI_API_KEY"]

def build(client, params):
    Path(params["out"]).write_text(json.dumps(dict(os.environ)))
    return Func(lambda: PROXY)
"""


def test_import_builds_its_tools_with_the_runs_environment_and_no_key_in_it(
    tmp_path, monkeypatch, app_home
):
    vendors = [vendor.key_env for vendor in VENDORS]
    claude_codes = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]
    for name in [*vendors, *claude_codes]:
        monkeypatch.setenv(name, KEY_IN_A_FRAME)
    # Another Library's custom server's key, in the shell and saved.
    monkeypatch.setenv("DEEP_REASONING_ELSEWHERE_KEY", "sk-elsewhere-in-the-shell")
    app_home.mkdir()
    (app_home / ".env").write_text("DEEP_REASONING_SAVED_KEY=sk-saved-for-elsewhere\n")
    monkeypatch.setenv("COPY_OF_A_KEY", KEY_IN_A_FRAME)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("CATALOG_DB", "catalog.sqlite")
    (tmp_path / "tools.py").write_text(READS_THE_ENVIRONMENT)
    out = tmp_path / "environment.json"
    config = tmp_path / "proxied.yaml"
    config.write_text(
        f"tools:\n  fetch: {{factory: build, factory_from: tools.py, out: {out}}}\n"
        "namespaces:\n  helper: {tools: [fetch]}\n"
    )
    cli.main(["import", str(config), "--library", str(tmp_path / "lib.sqlite")])
    seen = json.loads(out.read_text())
    assert {"HTTPS_PROXY", "CATALOG_DB", "PATH"} <= seen.keys()
    assert {seen[name] for name in vendors} == {NOT_A_KEY}
    gone = {*claude_codes, "DEEP_REASONING_ELSEWHERE_KEY", "COPY_OF_A_KEY"}
    assert not seen.keys() & gone


READS_OPENROUTERS_KEY = """\
import os
from deep_reasoner.primitives import Func

KEY = os.environ["OPENROUTER_API_KEY"]

def build(client, params):
    return Func(lambda: len(KEY))
"""


def test_import_refuses_a_tool_reading_a_key_only_some_chats_have(
    tmp_path, app_home, monkeypatch, no_vendor_keys
):
    (tmp_path / "router.py").write_text(READS_OPENROUTERS_KEY)
    config = tmp_path / "router.yaml"
    config.write_text(
        "tools:\n  ping: {factory: build, factory_from: router.py}\n"
        "namespaces:\n  helper: {tools: [ping]}\n"
    )
    import_it = ["import", str(config), "--library", str(tmp_path / "lib.sqlite")]
    refused = "Tool ping .*KeyError: 'OPENROUTER_API_KEY'"
    with pytest.raises(SystemExit, match=refused):
        cli.main(import_it)
    # Saved, it reaches only OpenRouter's chats.
    app_home.mkdir()
    (app_home / ".env").write_text("OPENROUTER_API_KEY=sk-router-saved-here\n")
    with pytest.raises(SystemExit, match=refused):
        cli.main(import_it)
    # In the shell, every chat's run has it.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router-in-the-shell")
    cli.main(import_it)


@pytest.mark.parametrize(
    ("where", "name"),
    [
        ("saved", "OPENROUTER_API_KEY"),
        ("saved", "DEEP_REASONING_ELSEWHERE_KEY"),
        ("shell", "GEMINI_API_KEY"),
        ("shell", "DEEP_REASONING_ELSEWHERE_KEY"),
        ("shell", "ANTHROPIC_API_KEY"),
        ("shell", "CLAUDE_CODE_OAUTH_TOKEN"),
    ],
)
def test_import_refuses_a_key_saved_in_settings_or_set_in_the_shell(
    tmp_path, app_home, monkeypatch, where, name
):
    if where == "saved":
        app_home.mkdir()
        (app_home / ".env").write_text(f"{name}={KEY_IN_A_FRAME}\n")
    else:
        monkeypatch.setenv(name, KEY_IN_A_FRAME)
    config = tmp_path / "keyed.yaml"
    config.write_text(f"namespaces:\n  helper: {{vars: {{key: {KEY_IN_A_FRAME}}}}}\n")
    with pytest.raises(SystemExit, match="Namespace helper holds a key"):
        cli.main(["import", str(config), "--library", str(tmp_path / "lib.sqlite")])


@pytest.mark.parametrize("command", [[], ["import", "CONFIG"]])
def test_a_library_in_a_folder_that_is_not_there_is_refused_and_nothing_made(
    served, tmp_path, advisors, free_port, command
):
    missing = tmp_path / "typo" / "lib.sqlite"
    argv = [str(advisors) if each == "CONFIG" else each for each in command]
    options = ["--port", str(free_port), "--no-browser", "--library", str(missing)]
    with pytest.raises(SystemExit, match=f"There is no folder {missing.parent}"):
        cli.main([*options, *argv])
    assert not missing.parent.exists()
    assert served == []


def unfinished_run(library):
    """A run going in the app file, as the first app's would be: started, not ended."""
    chat = create_chat(library)["id"]
    started = '{"seq": 1, "t": 1, "kind": "turn.start", "prompt": "Hi."}'
    with closing(open_library(library)) as connection:
        connection.execute("INSERT INTO runs VALUES ('a-run', ?)", (chat,))
        connection.execute("INSERT INTO events VALUES ('a-run', 1, ?)", (started,))


def events_of(library):
    with closing(open_library(library)) as connection:
        return connection.execute("SELECT seq, data FROM events").fetchall()


STARTED_NOT = (
    "so this one didn't start, as two on one Library would end each other's runs"
)


@pytest.mark.parametrize(
    ("named", "command", "refused"),
    [
        ("lib.sqlite", [], STARTED_NOT),
        ("link.sqlite", [], STARTED_NOT),
        ("relative", [], STARTED_NOT),
        ("lib.sqlite", ["import", "CONFIG"], "so nothing was imported"),
        ("lib.sqlite", ["reset", "--yes"], "so nothing was deleted"),
    ],
    ids=[
        "serve",
        "serve through a symlink",
        "serve by a relative path",
        "import",
        "reset",
    ],
)
def test_a_library_an_app_serves_is_refused_to_a_second_app_import_and_reset(
    monkeypatch, tmp_path, advisors, free_port, named, command, refused
):
    library = tmp_path / "lib.sqlite"
    (tmp_path / "link.sqlite").symlink_to(library)
    monkeypatch.chdir(tmp_path)
    given = "lib.sqlite" if named == "relative" else str(tmp_path / named)
    argv = [str(advisors) if each == "CONFIG" else each for each in command]
    second = ["--no-browser", "--port", "0", "--library", given, *argv]
    seen = []

    def serve(server, sockets):
        unfinished_run(library)
        before = events_of(library)
        with pytest.raises(SystemExit) as told:
            cli.main(second)
        assert told.value.code == (
            f"Deep Reasoning is already using {given} (serving, importing or resetting"
            f" it), {refused}: stop it, or wait for it to finish."
        )
        seen.append(events_of(library) == before and library.is_file())

    monkeypatch.setattr(uvicorn.Server, "run", serve)
    cli.main(["--port", str(free_port), "--no-browser", "--library", str(library)])
    assert seen == [True]


def test_an_app_started_while_a_reset_deletes_is_refused_and_makes_no_library(
    app_home, monkeypatch, served, free_port
):
    library = app_home / "library.sqlite"
    open_library(library).close()
    delete_all = cli.delete_all

    def deleting(paths):
        delete_all(paths)
        refused = "already using .*, so this one didn't start.*wait for it to finish"
        with pytest.raises(SystemExit, match=refused):
            cli.main(["--port", str(free_port), "--no-browser"])

    monkeypatch.setattr(cli, "delete_all", deleting)
    cli.main(["reset", "--yes"])
    assert not library.exists()
    assert served == []


def test_the_lock_is_let_go_when_the_app_stops_or_is_killed(served, tmp_path):
    library = tmp_path / "lib.sqlite"
    options = ["--port", "0", "--no-browser", "--library", str(library)]
    cli.main(options)
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_THE_LOCK, str(library)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline() == "held\n"
        with pytest.raises(SystemExit, match="already using"):
            cli.main(options)
    finally:
        holder.kill()
        holder.wait()
    cli.main(options)
    assert len(served) == 2


def test_a_file_system_without_locks_is_said_and_the_app_starts_all_the_same(
    served, monkeypatch, capsys, tmp_path, free_port
):
    def no_locks(lock, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    library = tmp_path / "lib.sqlite"
    cli.main(["--port", str(free_port), "--no-browser", "--library", str(library)])
    said = capsys.readouterr().out
    assert "can't lock it, so nothing stops a second Deep Reasoning" in said
    assert "No locks available" in said
    assert len(served) == 1


def test_a_file_that_is_not_a_library_exits_with_a_plain_error(
    served, tmp_path, free_port
):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a database\n")
    with pytest.raises(SystemExit, match="Could not open the Library at .*notes.txt"):
        cli.main(["--port", str(free_port), "--no-browser", "--library", str(notes)])
    assert served == []


class Terminal(io.StringIO):
    """Standard input from a person at a terminal, who types what it holds."""

    def isatty(self):
        return True


@pytest.fixture
def deleting_fails(monkeypatch):
    """Deleting anything fails the test, so a refusal gone wrong can't delete a home or /."""

    def deleted(path, *args, **kwargs):
        pytest.fail(f"deleted {path}")

    monkeypatch.setattr(shutil, "rmtree", deleted)
    monkeypatch.setattr(Path, "unlink", deleted)


def app_file(app_home, library):
    """A Library at *library* with its data folder, and what older versions left in
    *app_home*: the paths a reset deletes, but SQLite's side files."""
    open_library(library).close()
    data = data_folder(app_home, library)
    (data / "work" / "some-chat").mkdir(parents=True)
    (data / "runs" / "some-run").mkdir(parents=True)
    old_files = [app_home / "chats.sqlite", app_home / "settings.json"]
    for path in old_files:
        path.write_text("left by an older version\n")
    old_folders = [app_home / "runs", app_home / "work"]
    for folder in old_folders:
        (folder / "old").mkdir(parents=True)
    return [library, data, *old_files, *old_folders]


def as_made_by(version, library):
    """A Library file as *version* made it, without the tables of the versions after it."""
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


def side_files(library):
    return [library.with_name(library.name + side) for side in ("-wal", "-shm")]


def test_reset_yes_deletes_the_app_file_its_folders_and_old_leftovers(
    app_home, tmp_path
):
    library = app_home / "library.sqlite"
    project = tmp_path / "project"
    project.mkdir()
    create_chat(library, folder=str(project))
    doomed = app_file(app_home, library)
    for side_file in side_files(library):
        side_file.write_text("left by a crash\n")
    # Empty, as SQLite leaves no journal with a change in it but after a crash mid-change.
    journal = library.with_name(library.name + "-journal")
    journal.write_bytes(b"")
    doomed += [*side_files(library), journal]
    # An app that served the file left its lock, which a reset keeps: one starting may hold it.
    lock = library.with_name(library.name + ".lock")
    lock.write_bytes(b"")
    keys = app_home / ".env"
    keys.write_text(KEYS)
    other = data_folder(app_home, tmp_path / "other.sqlite")
    other.mkdir(parents=True)
    cli.main(["reset", "--yes"])
    assert [path for path in doomed if os.path.lexists(path)] == []
    assert lock.is_file()
    assert keys.read_text() == KEYS
    assert project.is_dir()
    assert other.is_dir()


@pytest.mark.parametrize("answer", ["no\n", "yes\n", "\n", "delete it\n", ""])
def test_reset_lists_what_goes_and_deletes_nothing_unless_told_delete(
    app_home, monkeypatch, capsys, answer
):
    library = app_home / "library.sqlite"
    doomed = app_file(app_home, library)
    monkeypatch.setattr(sys, "stdin", Terminal(answer))
    with pytest.raises(SystemExit, match="Nothing was deleted"):
        cli.main(["reset"])
    said = capsys.readouterr().out
    assert "~/.deep-reasoning/library.sqlite" in said
    assert "~/.deep-reasoning/chats.sqlite" in said
    # With no keys saved, none are said to be kept.
    assert ".env" not in said and "It keeps every folder a chat chose" in said
    assert all(os.path.lexists(path) for path in doomed)


def test_reset_says_it_keeps_the_keys_when_some_are_saved(
    app_home, monkeypatch, capsys
):
    library = app_home / "library.sqlite"
    app_file(app_home, library)
    keys = app_home / ".env"
    keys.write_text("OPENAI_API_KEY=sk-test-not-a-real-key\n")
    monkeypatch.setattr(sys, "stdin", Terminal("no\n"))
    with pytest.raises(SystemExit, match="Nothing was deleted"):
        cli.main(["reset"])
    said = capsys.readouterr().out
    assert "It keeps your keys, in ~/.deep-reasoning/.env" in said
    assert keys.is_file()


def test_typing_delete_resets(app_home, monkeypatch):
    library = app_home / "library.sqlite"
    doomed = app_file(app_home, library)
    monkeypatch.setattr(sys, "stdin", Terminal("delete\n"))
    cli.main(["reset"])
    assert [path for path in doomed if os.path.lexists(path)] == []


def test_reset_without_a_terminal_or_yes_deletes_nothing(app_home, monkeypatch):
    library = app_home / "library.sqlite"
    doomed = app_file(app_home, library)
    monkeypatch.setattr(sys, "stdin", io.StringIO("delete\n"))
    with pytest.raises(SystemExit, match="no terminal.*--yes"):
        cli.main(["reset"])
    assert all(os.path.lexists(path) for path in doomed)


@pytest.mark.parametrize("terminal", [True, False], ids=["answered no", "no terminal"])
def test_a_reset_that_deletes_nothing_leaves_and_lists_no_side_file_of_its_own(
    app_home, monkeypatch, capsys, terminal
):
    library = app_home / "library.sqlite"
    app_file(app_home, library)
    # In WAL mode, as the app leaves a Library it served: once closed, nothing beside it.
    with closing(sqlite3.connect(library)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
    monkeypatch.setattr(sys, "stdin", Terminal("no\n") if terminal else io.StringIO())
    with pytest.raises(SystemExit, match="(?i)nothing was deleted"):
        cli.main(["reset"])
    said = capsys.readouterr().out
    assert re.findall(r"-wal|-shm|\.lock", said) == []
    assert [path for path in side_files(library) if path.exists()] == []


def test_a_declined_reset_lists_and_leaves_only_the_side_files_already_there(
    app_home, monkeypatch, capsys
):
    library = app_home / "library.sqlite"
    app_file(app_home, library)
    with closing(sqlite3.connect(library)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
    wal, shm = side_files(library)
    wal.write_bytes(b"")
    monkeypatch.setattr(sys, "stdin", Terminal("no\n"))
    with pytest.raises(SystemExit, match="Nothing was deleted"):
        cli.main(["reset"])
    said = capsys.readouterr().out
    assert re.findall(r"-wal|-shm", said) == ["-wal"]
    assert (wal.exists(), shm.exists()) == (True, False)


@pytest.mark.parametrize("before", [True, False], ids=["before reset", "after reset"])
def test_reset_library_resets_that_file_alone_not_the_default_or_old_leftovers(
    app_home, tmp_path, before
):
    kept = app_file(app_home, app_home / "library.sqlite")
    shared = tmp_path / "shared.sqlite"
    open_library(shared).close()
    option = ["--library", str(shared)]
    cli.main([*option, "reset", "--yes"] if before else ["reset", "--yes", *option])
    assert not shared.exists()
    assert all(os.path.lexists(path) for path in kept)


def test_reset_with_nothing_there_says_so(app_home, capsys):
    app_home.mkdir()
    keys = app_home / ".env"
    keys.write_text(KEYS)
    cli.main(["reset"])
    assert "Nothing to reset" in capsys.readouterr().out
    assert keys.read_text() == KEYS


def test_reset_says_the_app_stopped_without_closing_the_file_when_the_wal_holds_changes(
    app_home, monkeypatch, capsys
):
    library = app_home / "library.sqlite"
    app_file(app_home, library)
    library.with_name(library.name + "-wal").write_text("a change\n")
    monkeypatch.setattr(sys, "stdin", Terminal("no\n"))
    with pytest.raises(SystemExit, match="Nothing was deleted"):
        cli.main(["reset"])
    assert "the app stopped without closing the file" in capsys.readouterr().out


def test_reset_never_deletes_through_a_symlink(app_home, tmp_path):
    library = app_home / "library.sqlite"
    app_file(app_home, library)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text("mine\n")
    data = data_folder(app_home, library)
    (data / "work" / "some-chat" / "link").symlink_to(outside)
    shutil.rmtree(app_home / "runs")
    (app_home / "runs").symlink_to(outside)
    cli.main(["reset", "--yes"])
    assert (outside / "notes.txt").read_text() == "mine\n"
    assert not data.exists()
    assert not os.path.lexists(app_home / "runs")


def test_reset_keeps_a_folder_holding_one_a_chat_chose(app_home):
    library = app_home / "library.sqlite"
    old_scratch = app_home / "work" / "old-chat"
    old_scratch.mkdir(parents=True)
    create_chat(library, folder=str(old_scratch))
    cli.main(["reset", "--yes"])
    assert old_scratch.is_dir()
    assert not library.exists()


@pytest.mark.parametrize("folder", ["", "projects"], ids=["the home", "a project"])
def test_reset_refuses_a_folder_as_the_library(deleting_fails, home, app_home, folder):
    target = home / folder
    target.mkdir(exist_ok=True)
    (target / "notes.txt").write_text("mine\n")
    app_home.mkdir()
    (app_home / ".env").write_text(KEYS)
    with pytest.raises(SystemExit, match="is not a file, so nothing was deleted"):
        cli.main(["reset", "--yes", "--library", str(target)])
    assert (target / "notes.txt").read_text() == "mine\n"
    assert (app_home / ".env").read_text() == KEYS


def test_reset_refuses_a_fifo_as_the_library(deleting_fails, tmp_path):
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    with pytest.raises(SystemExit, match="is not a file, so nothing was deleted"):
        cli.main(["reset", "--yes", "--library", str(pipe)])
    assert stat.S_ISFIFO(os.lstat(pipe).st_mode)


def test_reset_refuses_a_path_naming_no_file(deleting_fails):
    with pytest.raises(SystemExit, match="names no file, so nothing was deleted"):
        cli.main(["reset", "--yes", "--library", "/"])


def test_reset_refuses_a_symlinked_library_naming_its_target(
    deleting_fails, app_home, tmp_path
):
    real = tmp_path / "real.sqlite"
    open_library(real).close()
    data = data_folder(app_home, real)
    data.mkdir(parents=True)
    link = tmp_path / "link.sqlite"
    link.symlink_to(real)
    refused = f"symlink to {re.escape(str(real))}, so nothing was deleted.*--library"
    with pytest.raises(SystemExit, match=refused):
        cli.main(["reset", "--yes", "--library", str(link)])
    assert link.is_symlink()
    assert real.is_file()
    assert data.is_dir()


def test_reset_leaves_side_files_with_no_library_beside_them(tmp_path, capsys):
    diary = tmp_path / "diary"
    sides = [*side_files(diary), diary.with_name("diary-journal")]
    for side in sides:
        side.write_text("mine\n")
    cli.main(["reset", "--yes", "--library", str(diary)])
    assert "Nothing to reset" in capsys.readouterr().out
    assert all(side.read_text() == "mine\n" for side in sides)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_a_declined_reset_leaves_an_older_library_as_it_was(
    app_home, monkeypatch, version
):
    library = app_home / "library.sqlite"
    as_made_by(version, library)
    before = library.read_bytes()
    monkeypatch.setattr(sys, "stdin", Terminal("no\n"))
    with pytest.raises(SystemExit, match="Nothing was deleted"):
        cli.main(["reset"])
    assert library.read_bytes() == before


def another_apps_database(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE todos (task TEXT)")
        connection.commit()


@pytest.mark.parametrize(
    "make",
    [
        lambda path: path.write_bytes(b""),
        lambda path: path.write_text("not a database\n"),
        another_apps_database,
    ],
    ids=["empty", "text", "another app's database"],
)
def test_reset_refuses_a_file_that_is_not_a_library_and_leaves_it_as_it_was(
    tmp_path, make
):
    todo = tmp_path / "todo.txt"
    make(todo)
    before = todo.read_bytes()
    with pytest.raises(SystemExit, match="Could not read .*todo.txt.*nothing was"):
        cli.main(["reset", "--yes", "--library", str(todo)])
    assert todo.read_bytes() == before


def test_reset_refuses_a_library_a_crash_left_mid_change_and_says_how_to_mend_it(
    app_home,
):
    library = app_home / "library.sqlite"
    open_library(library).close()
    before = library.read_bytes()
    journal = library.with_name(library.name + "-journal")
    journal.write_text("a change part way through\n")
    with pytest.raises(SystemExit, match="nothing was deleted.*start the app once"):
        cli.main(["reset", "--yes"])
    assert library.read_bytes() == before
    assert journal.is_file()
