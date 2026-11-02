"""The `deep-reasoning` command: serve the app on 127.0.0.1 and open it in a tab, import a
config into the Library, or reset: delete the Library file and its folders."""

import argparse
import contextlib
import errno
import logging
import os
import shutil
import signal
import socket
import sqlite3
import sys
import threading
import webbrowser
from pathlib import Path

import structlog
import uvicorn
import yaml

from deep_reasoning.app import create_app
from deep_reasoning.app_tools import open_existing
from deep_reasoning.chats import chats_in
from deep_reasoning.folders import data_folder, holds_any, short_path
from deep_reasoning.library import (
    LIBRARY_ONLY,
    SCHEMA_VERSION,
    LibraryError,
    import_config,
    open_library,
    schema_version,
)
from deep_reasoning.settings import DEFAULT_FOLDER, ENV_FILE, settings_in

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_PORT = 65535
# The app's home: its runs, scratch folders and keys, and its app file by default.
HOME_DIR = ".deep-reasoning"
LIBRARY_FILE = "library.sqlite"
# SQLite's files beside an app file: while it is open, or after a crash.
WAL = "-wal"
SIDES = (WAL, "-shm", "-journal")
# What versions before the app file kept directly in the app's home, unread since.
LEFTOVERS = ("chats.sqlite", "settings.json", "runs", "work")
CONFIRMATION = "delete"
SIDE = "SQLite's file beside the Library"
THE_LIBRARY = "the Library: its namespaces, decompositions, tools, chats and settings"
ITS_FOLDERS = "the Library's run and scratch folders"
OLD = "left by an older version"
STOP_FIRST = (
    "Stop Deep Reasoning before a reset: deleting its Library while it runs breaks it."
)
BUSY = (
    "A run may be going, or the app stopped during one: "
    "the Library's -wal file holds changes."
)
# What a read-only connection meets in a file a crash left part way through a change.
MID_CHANGE = "SQLITE_READONLY_ROLLBACK"
CRASHED = " A crash left it part way through a change: start the app once, then reset."

log = structlog.get_logger()


class Server(uvicorn.Server):
    """uvicorn's server, ending the runs before it waits on open responses: a run's
    stream stays open until the run ends, so uvicorn would wait on it for good."""

    async def shutdown(self, sockets=None):
        await self.config.app.state.runs.close()
        await super().shutdown(sockets)


def listen(port):
    """A socket listening on HOST:port, or a plain error when the port is taken."""
    try:
        return socket.create_server((HOST, port))
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        sys.exit(
            f"Port {port} is already in use. Pick another with --port, "
            f"for example: deep-reasoning --port {port + 1}"
        )


def open_or_exit(path):
    """The Library at *path*, created if new, or a plain error when it can't be one."""
    try:
        return open_library(path)
    except (OSError, sqlite3.Error, LibraryError) as error:
        sys.exit(f"Could not open the Library at {path}: {error}")


def import_or_exit(library, config):
    with contextlib.closing(open_or_exit(library)) as connection:
        try:
            imported = import_config(connection, config)
        except (OSError, ValueError, yaml.YAMLError, sqlite3.Error) as error:
            sys.exit(f"Could not import {config}; the Library is unchanged.\n{error}")
    log.info(
        "Imported into the Library",
        config=str(config),
        library=str(library),
        **imported,
    )


def file_or_exit(library):
    """A plain error, deleting nothing, unless *library* is a file or nothing at all: a
    reset never deletes a folder, a device, or a file through a symlink."""
    if not library.name:
        sys.exit(f"{library} names no file, so nothing was deleted.")
    if library.is_symlink():
        sys.exit(
            f"{library} is a symlink to {os.path.realpath(library)}, so nothing was "
            "deleted. To reset that file, pass it with --library."
        )
    if os.path.lexists(library) and not library.is_file():
        sys.exit(f"{library} is not a file, so nothing was deleted.")


def chosen_in(library):
    """The folders the Library's chats and settings chose, read without writing the file;
    a LibraryError for a file that isn't a Library this version can open."""
    with contextlib.closing(open_existing(library, "ro")) as connection:
        version = schema_version(connection)
        if not LIBRARY_ONLY <= version <= SCHEMA_VERSION:
            raise LibraryError(
                "it is not a Library this version of Deep Reasoning can open"
            )
        if version == LIBRARY_ONLY:
            return []
        chosen = [chat["folder"] for chat in chats_in(connection)]
        chosen.append(settings_in(connection).get(DEFAULT_FOLDER))
    return [folder for folder in chosen if folder]


def chosen_or_exit(library):
    """chosen_in, or a plain error, deleting nothing."""
    try:
        return chosen_in(library)
    except (sqlite3.Error, LibraryError) as error:
        crashed = getattr(error, "sqlite_errorname", None) == MID_CHANGE
        sys.exit(
            f"Could not read the Library at {library}, so nothing was deleted: "
            f"{error}.{CRASHED if crashed else ''}"
        )


def reset_targets(library, home):
    """What a reset deletes, as (path, what it is), each that is there: SQLite's files
    beside the Library first, and only beside one; the older versions' leftovers only
    for the default Library, as theirs."""
    targets = []
    if library.is_file():
        targets += [(library.with_name(library.name + side), SIDE) for side in SIDES]
    targets += [(library, THE_LIBRARY), (data_folder(home, library), ITS_FOLDERS)]
    if library == home / LIBRARY_FILE:
        targets += [(home / name, OLD) for name in LEFTOVERS]
    return [(path, what) for path, what in targets if os.path.lexists(path)]


def reset_plan(doomed, kept, home, busy):
    width = max(len(short_path(path)) for path, _ in doomed)
    lines = [STOP_FIRST, *([BUSY] if busy else []), "", "This deletes:"]
    lines += [f"  {short_path(path):<{width}}  {what}" for path, what in doomed]
    lines += [
        f"It keeps {short_path(path)}, as a chat chose it or a folder in it."
        for path in kept
    ]
    lines += [
        f"It keeps your keys, in {short_path(home / ENV_FILE)} (remove them in settings),",
        "every folder a chat chose, and other Library files and their folders.",
    ]
    return "\n".join(lines)


def confirmed():
    """Whether the person at the terminal typed the confirmation; Ctrl+C or Ctrl+D is a no."""
    if not sys.stdin.isatty():
        sys.exit(
            "There is no terminal to ask on, so nothing was deleted. "
            "Pass --yes to reset without asking."
        )
    try:
        return input(f"Type {CONFIRMATION} to delete them: ").strip() == CONFIRMATION
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def delete(path):
    """Delete *path*, a folder with all it holds; a symlink only as itself, even one to a
    folder, so nothing is deleted through one."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def delete_all(paths):
    """Delete *paths* in turn, deaf to Ctrl+C meanwhile, as stopping part way would leave
    half a Library."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for path in paths:
            delete(path)
    except OSError as error:
        sys.exit(
            f"Could not delete {short_path(path)}: {error}. What is listed above it is "
            "deleted; it and what is below it are left, it perhaps in part."
        )
    finally:
        signal.signal(signal.SIGINT, previous)


def reset_or_exit(library, home, yes):
    """Delete the app file at *library* and what is its, once it has said what goes and
    what stays and, unless *yes*, been told to."""
    file_or_exit(library)
    # Before the Library is read: reading a Library in WAL mode makes an empty -wal.
    wal = library.with_name(library.name + WAL)
    busy = library.is_file() and os.path.isfile(wal) and os.path.getsize(wal) > 0
    chosen = chosen_or_exit(library) if library.is_file() else []
    there = reset_targets(library, home)
    kept = [path for path, _ in there if holds_any(path, chosen)]
    doomed = [(path, what) for path, what in there if path not in kept]
    if not doomed:
        print(
            f"Nothing to reset: no Library at {short_path(library)}, nor folders of it."
        )
        return
    print(reset_plan(doomed, kept, home, busy))
    if not yes and not confirmed():
        sys.exit("Nothing was deleted.")
    delete_all([path for path, _ in doomed])
    print(
        "Deleted them. The next start makes a fresh Library, with the historian and "
        "librarian namespaces."
    )


def serve(port, no_browser, library, home):
    open_or_exit(library).close()
    server = Server(uvicorn.Config(create_app(library, home)))
    with listen(port) as sock:
        url = f"http://{HOST}:{sock.getsockname()[1]}/"
        log.info("Deep Reasoning is running", url=url, library=str(library))
        if not no_browser:
            # The tab's request waits for uvicorn; a thread, as some launchers block.
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        # uvicorn re-raises Ctrl+C after a clean shutdown; uvicorn.run hides it too.
        with contextlib.suppress(KeyboardInterrupt):
            server.run(sockets=[sock])


def main(argv=None):
    # Importing deep_reasoner quiets structlog to warnings; this command's lines are info.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
    )
    # Tracebacks without each frame's variables: a frame may hold a key.
    structlog.dev.ConsoleRenderer.get_active().exception_formatter = (
        structlog.dev.plain_traceback
    )
    parser = argparse.ArgumentParser(
        description="Start Deep Reasoning and open it in a browser tab."
    )
    home = Path.home() / HOME_DIR
    default_library = home / LIBRARY_FILE
    library_help = f"the Library file to use (default {default_library})"
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"the port to serve on (default {DEFAULT_PORT}; 0 picks a free one)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="don't open a browser tab"
    )
    parser.add_argument(
        "--library", type=Path, default=default_library, help=library_help
    )
    commands = parser.add_subparsers(dest="command", metavar="command")
    importer = commands.add_parser(
        "import",
        help="read a deep_reasoner config into the Library, then exit",
        description="Read a deep_reasoner config into the Library, replacing same-named items.",
    )
    importer.add_argument("config", type=Path, help="the config's main YAML file")
    resetter = commands.add_parser(
        "reset",
        help="delete the Library file and its run and scratch folders, after asking",
        description=(
            "Delete the Library file (its namespaces, decompositions, tools, chats and "
            "settings), its runs' and chats' scratch folders, and what older versions "
            "left in the app's home, after asking. Your keys and the folders chats chose "
            "stay. Stop the app first."
        ),
    )
    resetter.add_argument("--yes", action="store_true", help="delete without asking")
    for command in (importer, resetter):
        # SUPPRESS keeps a --library given before the command from being reset to the default.
        command.add_argument(
            "--library", type=Path, default=argparse.SUPPRESS, help=library_help
        )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= MAX_PORT:
        parser.error(f"--port must be between 0 and {MAX_PORT}")
    if args.command == "import":
        import_or_exit(args.library, args.config)
        return
    if args.command == "reset":
        reset_or_exit(args.library, home, args.yes)
        return
    serve(args.port, args.no_browser, args.library, home)
