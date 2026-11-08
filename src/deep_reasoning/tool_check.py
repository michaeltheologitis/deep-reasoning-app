"""Build tools as a run builds them, in a process of their own, to tell which don't
before anything grants them: one that doesn't build stops every run that declares it.

check_tools(tools, config, env, work) is the interface, for the library tool and for an
import alike. *tools* are by name, each its config block and its Python source (the
block's factory_from is set to the file the source is written to); *config* is a run's
engine config as V2Config reads it, whose client, model, models presets and llm_kwargs
the tools are built with, exactly as the engine's build_reasoner builds them (inside
the run's LLM.alias, with model=); *env* is the environment they are built in, keys
included, as a run's worker has it; *work* is a folder for the check's files, each
check's deleted after. It returns a verdict per tool: None when it builds, else why
not, in words, naming its file only by its name.

Each check runs `python -P -m deep_reasoning.tool_check` from an empty folder, so the
tools see no chat's files or modules, in a process group of its own that is ended
after, so nothing a tool starts outlives it, and is stopped past BUILD_TIMEOUT_S. The
parent names the file the verdicts go to before any tool's code runs; the child notes
each tool's name in it before building it and its verdict after, and ends itself
whatever happens, so a thread a tool left can't hold it up. The tools' files are in a
folder of their own, apart from that file, so a tool that writes a file of that name
beside its own, or in its working folder, changes no verdict (one that seeks the file
out could forge one: it is the user's own code, unsandboxed in every run anyway). A tool
that ends or hangs the check is the one blamed, and the tools after it are checked again
without it.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml
from deep_reasoner import build_client
from deep_reasoner.tools import ToolBuildError
from deep_reasoner.v2 import LLM
from deep_reasoner.v2.cli import V2Config, make_tool

from deep_reasoning.library import dump, tool_file

YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
# A factory that takes longer is refused: every run would wait on it.
BUILD_TIMEOUT_S = 60
# What a check that ended without a verdict shows of what it printed.
SHOWN_OUTPUT_CHARS = 2_000
# The check's folder holds the tools' files in a folder of their own.
TOOLS = "tools"
Tools = Mapping[str, tuple[dict[str, Any], str]]


def config_of(spec: Path) -> dict[str, Any]:
    """The engine config of the run whose spec.yaml is *spec*: a run's worker always has
    one, so a missing one is a bug and raises."""
    return yaml.load(spec.read_text(), Loader=YAML_LOADER)["config"]


def check_tools(
    tools: Tools, config: dict[str, Any], env: Mapping[str, str], work: Path
) -> dict[str, str | None]:
    """Why each of *tools* doesn't build in a run of *config*, in *env*: None for each
    that does. A tool that ends or hangs a check is blamed, and the rest checked again."""
    verdicts = {}
    while pending := {
        name: each for name, each in tools.items() if name not in verdicts
    }:
        verdicts |= checked_once(pending, config, env, work)
    return verdicts


def checked_once(
    tools: Tools, config: dict[str, Any], env: Mapping[str, str], work: Path
) -> dict[str, str | None]:
    """The verdicts one check of *tools* reached: at least one, as a check that ended
    early blames the tool it was building, or every one when it built none."""
    work.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=work))
    files = folder / TOOLS
    files.mkdir()
    try:
        blocks = {}
        for name, (block, source) in tools.items():
            file = files / tool_file(name)
            file.write_text(source, encoding="utf-8")
            blocks[name] = block | {"factory_from": str(file)}
        notes, printed, timed_out = run(blocks, config, env, folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    verdicts = {note["built"]: note["verdict"] for note in notes if "built" in note}
    building = [note["building"] for note in notes if "building" in note]
    failed = [note["failed"] for note in notes if "failed" in note]
    if timed_out:
        why = f"it didn't build within {BUILD_TIMEOUT_S:g} seconds"
    elif failed:
        why = f"the check couldn't start: {failed[0]}"
    else:
        tail = printed[-SHOWN_OUTPUT_CHARS:].strip()
        why = "building it ended the check" + (f", printing: {tail}" if tail else "")
    if building and building[-1] not in verdicts:
        verdicts[building[-1]] = why
    if not verdicts:
        verdicts = dict.fromkeys(tools, why)
    # Each tool's file by its name alone: the folder it was built in is gone.
    shown = f"{files}{os.sep}"
    return {
        name: None if why is None else why.replace(shown, "")
        for name, why in verdicts.items()
    }


def run(
    blocks: dict[str, dict[str, Any]],
    config: dict[str, Any],
    env: Mapping[str, str],
    folder: Path,
) -> tuple[list[dict[str, Any]], str, bool]:
    """The child, run on *blocks* from an empty folder in *folder*, in a process group of
    its own that is ended after: what it noted, what it printed, and whether it ran out
    of time."""
    empty, printed, notes = folder / "cwd", folder / "printed.txt", folder / "notes"
    empty.mkdir()
    notes.touch()
    asked = {"tools": blocks, "config": config, "notes": str(notes)}
    timed_out = False
    with printed.open("w") as output:
        check = subprocess.Popen(
            [sys.executable, "-P", "-m", __name__],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=output,
            cwd=empty,
            env=dict(env),
            start_new_session=True,
            text=True,
        )
        try:
            check.communicate(dump(asked), BUILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Only while it is unreaped, so its pid, and its group's, is no one else's;
            # once it has ended, it has ended its group itself.
            if check.returncode is None:
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(check.pid, signal.SIGKILL)
            check.wait()
    return read_notes(notes), printed.read_text(errors="replace"), timed_out


def read_notes(notes: Path) -> list[dict[str, Any]]:
    """The child's notes, each a JSON line; one cut short as it was killed is left out."""
    found = []
    for line in notes.read_text(errors="replace").splitlines():
        with suppress(json.JSONDecodeError):
            found.append(json.loads(line))
    return found


def note(notes: Path, entry: dict[str, Any]) -> None:
    with notes.open("a") as file:
        file.write(json.dumps(entry) + "\n")


def verdict(
    name: str, block: dict[str, Any], client: Any, config: V2Config
) -> str | None:
    """Why tool *name* doesn't build as a run of *config* builds it: None when it does."""
    try:
        make_tool(
            name,
            block,
            client=client,
            config_path=config.config_path,
            model=config.model,
        )
    except ToolBuildError as error:
        return error.detail
    except SystemExit as error:
        # The engine words what a factory raises, but a sys.exit() is no Exception.
        return f"it called sys.exit({error.code!r})"
    return None


def build_all(asked: dict[str, Any], notes: Path) -> None:
    """Build each tool *asked* holds as build_reasoner builds a run's: with the run's
    client, inside its LLM.alias, so a models preset resolves as in the run."""
    config = V2Config.model_validate(asked["config"])
    client = build_client(config.client)
    with LLM.alias(names=config.models, client=client, **config.llm_kwargs):
        for name, block in asked["tools"].items():
            note(notes, {"building": name})
            note(
                notes, {"built": name, "verdict": verdict(name, block, client, config)}
            )


def main() -> None:
    notes = None
    try:
        asked = yaml.load(sys.stdin, Loader=YAML_LOADER)
        notes = Path(asked["notes"])
        build_all(asked, notes)
    except Exception as error:
        if notes is not None:
            note(notes, {"failed": f"{type(error).__name__}: {error}"})
        raise
    finally:
        # Whatever happened, and whatever thread or process a tool left running: its
        # group, which this process leads when the check starts it, goes with it.
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(0)


if __name__ == "__main__":
    main()
